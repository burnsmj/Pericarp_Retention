### Stained Pericarp Mass Analysis
### Michael Burns
### 4/13/22

# Purpose: To determine whether or not the stain used in pericarp analysis provides
#          a significant mass.  Samples have been cooked in pairs with one pair being 
#          stained and the other not.  Both samples have their pericarp removed, dried,
#          and weighed.  A paired statistical analysis will be done to determine if stained
#          pericarp can be used in future analysis, or if unstained will also need to be 
#          collected.

# Libraries
library(tidyverse)
library(readxl)
library(magrittr)

# Data
data = read_xlsx('../Pericarp/Stained_Pericarp_Mass_Experiment.xlsx')

# Check for t-test vairance assumption - Looks good!
data %>%
  group_by(Stained) %>%
  summarise(mean_mass = mean(Average_Kernel_Pericarp_Mass),
            var_mass = var(Average_Kernel_Pericarp_Mass))

# Check for normality assumption - Looks okay. QQ plot tails are kind of far off
data %>%
  pivot_wider(id_cols = c(Sample_ID), values_from = Average_Kernel_Pericarp_Mass, names_from = Stained) %>%
  mutate(diffs = Y - N) %>%
  ggplot(aes(x = diffs))+
  geom_histogram(bins = 5)+
  theme_classic()

data %>%
  pivot_wider(id_cols = c(Sample_ID), values_from = Average_Kernel_Pericarp_Mass, names_from = Stained) %>%
  mutate(diffs = Y - N) %>%
  ggplot(aes(sample = diffs))+
  geom_qq()+
  geom_qq_line()+
  theme_classic()

# Check for outliers assumption - Looks okay.  Three possible outliers present.
data %>%
  pivot_wider(id_cols = c(Sample_ID), values_from = Average_Kernel_Pericarp_Mass, names_from = Stained) %>%
  mutate(diffs = Y - N) %>%
  ggplot(aes(y = diffs))+
  geom_boxplot(show.legend = F)+
  theme_classic()

# Remove outliers and test difference in groups with a t-test - Significantly different with p = 4.611e-05
data %>%
  pivot_wider(id_cols = c(Sample_ID), values_from = Average_Kernel_Pericarp_Mass, names_from = Stained) %>%
  mutate(diffs = Y - N) %>%
  mutate(diffs_q1 = quantile(diffs, 0.25),
         diffs_q3 = quantile(diffs, 0.75),
         diffs_iqr = diffs_q3 - diffs_q1,
         lower_thresh = diffs_q1 - (1.5*diffs_iqr),
         upper_thresh = diffs_q3 + (1.5*diffs_iqr)) %>%
  filter(diffs > lower_thresh & diffs < upper_thresh) %$%
  t.test(x = N, y = Y, paired = T, var.equal = T)
