### Pericarp Cook Test Progress and Quality Check

# Libraries
library(lubridate)
library(tidyverse)
library(lme4)

####################
# Benchtop Cook QC #
####################
# Data
data = read_csv('~/Downloads/Pericarp_Retention_Methods_Validation - Benchtop_Cook.csv') %>%
  select(Sample_ID, Hotplate_ID, Cook, Cook_Date, Peeler, pH, Cook_Time, Bath_Start, Bath_End, Dry_Pericarp_Mass_1, Dry_Pericarp_Mass_2, Pericarp_Mass_Avg, Dry_Kernels_Mass_1, Dry_Kernels_Mass_2) %>%
  filter(!is.na(Pericarp_Mass_Avg),
         Pericarp_Mass_Avg > 0) %>%
  mutate(Dry_Kernels_Mass_1 = case_when(Dry_Kernels_Mass_1 < 500 ~ NA,
                                        TRUE ~ Dry_Kernels_Mass_1),
         Dry_Kernels_Mass_2 = case_when(Dry_Kernels_Mass_2 < 500 ~ NA,
                                        TRUE ~ Dry_Kernels_Mass_2),
         Dry_Pericarp_Mass_Norm_1 = Dry_Pericarp_Mass_1 / (Dry_Kernels_Mass_1 / 1000),
         Dry_Pericarp_Mass_Norm_2 = Dry_Pericarp_Mass_2 / (Dry_Kernels_Mass_2 / 1000),
         Pericarp_Mass_Avg_Norm = case_when(!is.na(Dry_Pericarp_Mass_Norm_1) & !is.na(Dry_Pericarp_Mass_Norm_2) ~ (Dry_Pericarp_Mass_Norm_1 + Dry_Pericarp_Mass_Norm_2) / 2,
                                            !is.na(Dry_Pericarp_Mass_Norm_1) & is.na(Dry_Pericarp_Mass_Norm_2) ~ Dry_Pericarp_Mass_Norm_1,
                                            is.na(Dry_Pericarp_Mass_Norm_1) & !is.na(Dry_Pericarp_Mass_Norm_2) ~ Dry_Pericarp_Mass_Norm_2))

# Number of samples cooked so far
paste('Number of samples cooked:', nrow(data))
# Correlation of subsamples
paste('Spearman Correlation of Subsamples:', signif(cor(data$Dry_Pericarp_Mass_Norm_1,
                                                        data$Dry_Pericarp_Mass_Norm_2,
                                                        method = 'spearman',
                                                        use = 'complete.obs'),
                                                    3))
paste('Pearson Correlation of Subsamples:', signif(cor(data$Dry_Pericarp_Mass_Norm_1,
                                                       data$Dry_Pericarp_Mass_Norm_2,
                                                       use = 'complete.obs'),
      3))

# Check normality of data
normality_pval = shapiro.test(data$Pericarp_Mass_Avg_Norm)$p.value

# Announce results of normality check
trans_needed = FALSE
if(normality_pval < 0.05){
  print(paste('Pericarp Mass is not normally distributed with a p-value of:', normality_pval))
  trans_needed = TRUE
} else {
  print(paste('Pericarp Mass is normally distributed'))
}

# Perform a normalization transformation if the data is not normally distributed
if(trans_needed){
  # Check if new data is normally distributed
  trans_norm_pval = shapiro.test(log(data$Pericarp_Mass_Avg_Norm))$p.value
  
  # Announce results of normality check
  if(trans_norm_pval < 0.05){
    stop(paste('Log transformed Pericarp Mass is not normally distributed with a p-value of:', trans_norm_pval))
  } else {
    print(paste('Log transformed Pericarp Mass is normally distributed'))
    
    # Transform Data to Log-Based
    data = data %>%
      mutate(Pericarp_Mass_Avg_Norm = log(Pericarp_Mass_Avg_Norm),
             Dry_Pericarp_Mass_Norm_1 = log(Dry_Pericarp_Mass_Norm_1),
             Dry_Pericarp_Mass_Norm_2 = log(Dry_Pericarp_Mass_Norm_2))
  }
}

# Plot the data
plot(data$Dry_Pericarp_Mass_Norm_1,
     data$Dry_Pericarp_Mass_Norm_2,
     xlab = if(trans_needed) 'log(Subsample 1)' else 'Subsample 1',
     ylab = if(trans_needed) 'log(Subsample 2)' else 'Subsample 2',
     main = 'Subsample Correlation')
abline(lm(data$Dry_Pericarp_Mass_Norm_2 ~ data$Dry_Pericarp_Mass_Norm_1),
       col = 'red')

# Number of samples cooked in replicate so far
replicated_data = data %>%
  group_by(Sample_ID) %>% 
  filter(n() > 1) %>%
  select(Sample_ID, Hotplate_ID, Pericarp_Mass_Avg_Norm) %>%
  pivot_wider(names_from = Hotplate_ID, values_from = Pericarp_Mass_Avg_Norm)

# Plot the data
plot(replicated_data$A,
     replicated_data$B,
     xlab = if(trans_needed) 'log(Replicate A)' else 'Replicate A',
     ylab = if(trans_needed) 'log(Replicate B)' else 'Replicate B',
     main = 'Replicate Correlation')
abline(lm(replicated_data$B ~ replicated_data$A),
       col = 'red')

# Summary Table:
summary_table = tibble(Sample_Type = c('Subsample',
                                       'Replicate'),
                       N = c(nrow(data),
                             nrow(replicated_data)),
                       Spearman_R = c(signif(cor(data$Dry_Pericarp_Mass_Norm_1,
                                                 data$Dry_Pericarp_Mass_Norm_2,
                                                 method = 'spearman',
                                                 use = 'complete.obs'), 3),
                                      signif(cor(replicated_data$A,
                                                 replicated_data$B,
                                                 method = 'spearman',
                                                 use = 'complete.obs'), 3)),
                       Pearson_R = c(signif(cor(data$Dry_Pericarp_Mass_Norm_1,
                                                data$Dry_Pericarp_Mass_Norm_2,
                                                use = 'complete.obs'), 3),
                                     signif(cor(replicated_data$A,
                                                replicated_data$B,
                                                use = 'complete.obs'), 3)),
                       T_Test_P_Value = c(signif(t.test(data$Dry_Pericarp_Mass_Norm_1,
                                                        data$Dry_Pericarp_Mass_Norm_2,
                                                        paired = TRUE)$p.value, 3),
                                          signif(t.test(replicated_data$A,
                                                        replicated_data$B,
                                                        paired = TRUE)$p.value, 3)),
                       Log_Transformed = trans_needed)
                                       
# Create a mixed linear model to determine the proportion of variance explained by variance components
data_for_model = data %>%
  mutate(cook_time = as.numeric((Bath_Start - Cook_Time) / 60),
         steep_time = as.numeric(((Bath_End + (24*60*60)) - Bath_Start) / (60*60))) %>%
  select(-Cook_Time, -Bath_Start, -Bath_End)

# Check model assumptions of a and b hotplates being similar
model = lm(A ~ B, data = replicated_data)
opar = par(mfrow = c(2,2))
print(plot(model, which = 1:4))

# Sample ID is a combination of genotype and year information that is not broken out,
# Cook Date is to determine if there is a strong date effect that could explain the lack of correlation between replicates while subsample correlation is decent
# Cook time, steep time, and pH are to determine what proportion of variance the cooking parameters have on the pericarp mass
# Peeler is to ensure that we are not seeing a strong difference between the two people peeling samples.
model = lmer(Pericarp_Mass_Avg_Norm ~ (1|Sample_ID) + (1|Hotplate_ID) + (1|Cook_Date) + (1|cook_time) + (1|steep_time) + (1|pH) + (1|Peeler), data = data_for_model)

# Summarise Findings:
if(trans_needed){
  paste('Log transformed Pericarp Mass is not normally distributed so a log transformation was applied')
}

summary(model)$varcor %>%
  as_tibble() %>%
  mutate(Total_SS = sum(vcov),
         PVE = vcov / Total_SS) %>%
  select(grp, PVE) %>%
  rename(Variable = grp)

print(summary_table)

#################
# Rapid Cook QC #
#################
# Data
data = read_csv('~/Downloads/Pericarp_Retention_Methods_Validation - Rapid_Cook.csv') %>%
  select(Sample_ID, Rep, Cook, Cook_Date, Peeler, pH, Cook_Start, Cook_End, Dry_Pericarp_Mass_1, Dry_Pericarp_Mass_2, Pericarp_Mass_Avg, Dry_Kernels_Mass_1, Dry_Kernels_Mass_2) %>%
  filter(!is.na(Pericarp_Mass_Avg),
         Pericarp_Mass_Avg > 0) %>%
  mutate(Dry_Kernels_Mass_1 = case_when(Dry_Kernels_Mass_1 < 500 ~ NA,
                                        TRUE ~ Dry_Kernels_Mass_1),
         Dry_Kernels_Mass_2 = case_when(Dry_Kernels_Mass_2 < 500 ~ NA,
                                        TRUE ~ Dry_Kernels_Mass_2),
         Dry_Pericarp_Mass_Norm_1 = Dry_Pericarp_Mass_1 / (Dry_Kernels_Mass_1 / 1000),
         Dry_Pericarp_Mass_Norm_2 = Dry_Pericarp_Mass_2 / (Dry_Kernels_Mass_2 / 1000),
         Pericarp_Mass_Avg_Norm = case_when(!is.na(Dry_Pericarp_Mass_Norm_1) & !is.na(Dry_Pericarp_Mass_Norm_2) ~ (Dry_Pericarp_Mass_Norm_1 + Dry_Pericarp_Mass_Norm_2) / 2,
                                            !is.na(Dry_Pericarp_Mass_Norm_1) & is.na(Dry_Pericarp_Mass_Norm_2) ~ Dry_Pericarp_Mass_Norm_1,
                                            is.na(Dry_Pericarp_Mass_Norm_1) & !is.na(Dry_Pericarp_Mass_Norm_2) ~ Dry_Pericarp_Mass_Norm_2))

# Number of samples cooked so far
paste('Number of samples cooked:', nrow(data))
# Correlation of subsamples
paste('Spearman Correlation of Subsamples:', signif(cor(data$Dry_Pericarp_Mass_Norm_1,
                                                        data$Dry_Pericarp_Mass_Norm_2,
                                                        method = 'spearman',
                                                        use = 'complete.obs'),
                                                    3))
paste('Pearson Correlation of Subsamples:', signif(cor(data$Dry_Pericarp_Mass_Norm_1,
                                                       data$Dry_Pericarp_Mass_Norm_2,
                                                       use = 'complete.obs'),
                                                   3))

# Check normality of data
normality_pval = shapiro.test(data$Pericarp_Mass_Avg_Norm)$p.value

# Announce results of normality check
trans_needed = FALSE
if(normality_pval < 0.05){
  print(paste('Pericarp Mass is not normally distributed with a p-value of:', normality_pval))
  trans_needed = TRUE
} else {
  print(paste('Pericarp Mass is normally distributed'))
}

# Perform a normalization transformation if the data is not normally distributed
if(trans_needed){
  # Check if new data is normally distributed
  trans_norm_pval = shapiro.test(log(data$Pericarp_Mass_Avg_Norm))$p.value
  
  # Announce results of normality check
  if(trans_norm_pval < 0.05){
    stop(paste('Log transformed Pericarp Mass is not normally distributed with a p-value of:', trans_norm_pval))
  } else {
    print(paste('Log transformed Pericarp Mass is normally distributed'))
    
    # Transform Data to Log-Based
    data = data %>%
      mutate(Pericarp_Mass_Avg_Norm = log(Pericarp_Mass_Avg_Norm),
             Dry_Pericarp_Mass_Norm_1 = log(Dry_Pericarp_Mass_Norm_1),
             Dry_Pericarp_Mass_Norm_2 = log(Dry_Pericarp_Mass_Norm_2))
  }
}

# Plot the data
plot(data$Dry_Pericarp_Mass_Norm_1,
     data$Dry_Pericarp_Mass_Norm_2,
     xlab = if(trans_needed) 'log(Subsample 1)' else 'Subsample 1',
     ylab = if(trans_needed) 'log(Subsample 2)' else 'Subsample 2',
     main = 'Subsample Correlation')
abline(lm(data$Dry_Pericarp_Mass_Norm_2 ~ data$Dry_Pericarp_Mass_Norm_1),
       col = 'red')

# Number of samples cooked in replicate so far
replicated_data = data %>%
  group_by(Sample_ID) %>% 
  filter(n() > 1) %>%
  select(Sample_ID, Rep, Pericarp_Mass_Avg_Norm) %>%
  pivot_wider(names_from = Rep, values_from = Pericarp_Mass_Avg_Norm)

# Plot the data
plot(replicated_data$A,
     replicated_data$B,
     xlab = if(trans_needed) 'log(Replicate A)' else 'Replicate A',
     ylab = if(trans_needed) 'log(Replicate B)' else 'Replicate B',
     main = 'Replicate Correlation')
abline(lm(replicated_data$B ~ replicated_data$A),
       col = 'red')

# Summary Table:
summary_table = tibble(Sample_Type = c('Subsample',
                                       'Replicate'),
                       N = c(nrow(data),
                             nrow(replicated_data)),
                       Spearman_R = c(signif(cor(data$Dry_Pericarp_Mass_Norm_1,
                                                 data$Dry_Pericarp_Mass_Norm_2,
                                                 method = 'spearman',
                                                 use = 'complete.obs'), 3),
                                      signif(cor(replicated_data$A,
                                                 replicated_data$B,
                                                 method = 'spearman',
                                                 use = 'complete.obs'), 3)),
                       Pearson_R = c(signif(cor(data$Dry_Pericarp_Mass_Norm_1,
                                                data$Dry_Pericarp_Mass_Norm_2,
                                                use = 'complete.obs'), 3),
                                     signif(cor(replicated_data$A,
                                                replicated_data$B,
                                                use = 'complete.obs'), 3)),
                       T_Test_P_Value = c(signif(t.test(data$Dry_Pericarp_Mass_Norm_1,
                                                        data$Dry_Pericarp_Mass_Norm_2,
                                                        paired = TRUE)$p.value, 3),
                                          signif(t.test(replicated_data$A,
                                                        replicated_data$B,
                                                        paired = TRUE)$p.value, 3)),
                       Log_Transformed = trans_needed)

# Create a mixed linear model to determine the proportion of variance explained by variance components
data_for_model = data %>%
  mutate(cook_time = as.numeric((Cook_End - Cook_Start) / 60)) %>%
  select(-Cook_Start, -Cook_End)

# Check model assumptions of a and b hotplates being similar
model = lm(A ~ B, data = replicated_data)
opar = par(mfrow = c(2,2))
print(plot(model, which = 1:4))
dev.off()
# Sample ID is a combination of genotype and year information that is not broken out,
# Cook Date is to determine if there is a strong date effect that could explain the lack of correlation between replicates while subsample correlation is decent
# Cook time, steep time, and pH are to determine what proportion of variance the cooking parameters have on the pericarp mass
# Peeler is to ensure that we are not seeing a strong difference between the two people peeling samples.
model = lmer(Pericarp_Mass_Avg_Norm ~ (1|Sample_ID) + (1|Rep) + (1|Cook_Date) + (1|pH) + (1|Peeler), data = data_for_model)

# Summarise Findings:
if(trans_needed){
  paste('Log transformed Pericarp Mass is not normally distributed so a log transformation was applied')
}

summary(model)$varcor %>%
  as_tibble() %>%
  mutate(Total_SS = sum(vcov),
         PVE = vcov / Total_SS) %>%
  select(grp, PVE) %>%
  rename(Variable = grp)

print(summary_table)

#################
# Raw Kernel QC #
#################
# Data
data = read_csv('~/Downloads/Pericarp_Retention_Methods_Validation - Initial_Pericarp.csv') %>%
  select(Sample_ID, Rep, Date_Peel, Peeler, Dry_Pericarp_Mass_1, Dry_Pericarp_Mass_2, Pericarp_Mass_Avg, Dry_Kernels_Mass_1, Dry_Kernels_Mass_2) %>%
  filter(!is.na(Pericarp_Mass_Avg),
         Pericarp_Mass_Avg > 0) %>%
  mutate(Dry_Kernels_Mass_1 = case_when(Dry_Kernels_Mass_1 < 500 ~ NA,
                                        TRUE ~ Dry_Kernels_Mass_1),
         Dry_Kernels_Mass_2 = case_when(Dry_Kernels_Mass_2 < 500 ~ NA,
                                        TRUE ~ Dry_Kernels_Mass_2),
         Dry_Pericarp_Mass_Norm_1 = Dry_Pericarp_Mass_1 / (Dry_Kernels_Mass_1 / 1000),
         Dry_Pericarp_Mass_Norm_2 = Dry_Pericarp_Mass_2 / (Dry_Kernels_Mass_2 / 1000),
         Pericarp_Mass_Avg_Norm = case_when(!is.na(Dry_Pericarp_Mass_Norm_1) & !is.na(Dry_Pericarp_Mass_Norm_2) ~ (Dry_Pericarp_Mass_Norm_1 + Dry_Pericarp_Mass_Norm_2) / 2,
                                            !is.na(Dry_Pericarp_Mass_Norm_1) & is.na(Dry_Pericarp_Mass_Norm_2) ~ Dry_Pericarp_Mass_Norm_1,
                                            is.na(Dry_Pericarp_Mass_Norm_1) & !is.na(Dry_Pericarp_Mass_Norm_2) ~ Dry_Pericarp_Mass_Norm_2))

# Number of samples cooked so far
paste('Number of samples cooked:', nrow(data))
# Correlation of subsamples
paste('Spearman Correlation of Subsamples:', signif(cor(data$Dry_Pericarp_Mass_Norm_1,
                                                        data$Dry_Pericarp_Mass_Norm_2,
                                                        method = 'spearman',
                                                        use = 'complete.obs'),
                                                    3))
paste('Pearson Correlation of Subsamples:', signif(cor(data$Dry_Pericarp_Mass_Norm_1,
                                                       data$Dry_Pericarp_Mass_Norm_2,
                                                       use = 'complete.obs'),
                                                   3))

# Check normality of data
normality_pval = shapiro.test(data$Pericarp_Mass_Avg_Norm)$p.value

# Announce results of normality check
trans_needed = FALSE
if(normality_pval < 0.05){
  print(paste('Pericarp Mass is not normally distributed with a p-value of:', normality_pval))
  trans_needed = TRUE
} else {
  print(paste('Pericarp Mass is normally distributed'))
}

# Perform a normalization transformation if the data is not normally distributed
if(trans_needed){
  # Check if new data is normally distributed
  trans_norm_pval = shapiro.test(log(data$Pericarp_Mass_Avg_Norm))$p.value
  
  # Announce results of normality check
  if(trans_norm_pval < 0.05){
    stop(paste('Log transformed Pericarp Mass is not normally distributed with a p-value of:', trans_norm_pval))
  } else {
    print(paste('Log transformed Pericarp Mass is normally distributed'))
    
    # Transform Data to Log-Based
    data = data %>%
      mutate(Pericarp_Mass_Avg_Norm = log(Pericarp_Mass_Avg_Norm),
             Dry_Pericarp_Mass_Norm_1 = log(Dry_Pericarp_Mass_Norm_1),
             Dry_Pericarp_Mass_Norm_2 = log(Dry_Pericarp_Mass_Norm_2))
  }
}

# Plot the data
plot(data$Dry_Pericarp_Mass_Norm_1,
     data$Dry_Pericarp_Mass_Norm_2,
     xlab = if(trans_needed) 'log(Subsample 1)' else 'Subsample 1',
     ylab = if(trans_needed) 'log(Subsample 2)' else 'Subsample 2',
     main = 'Subsample Correlation')
abline(lm(data$Dry_Pericarp_Mass_Norm_2 ~ data$Dry_Pericarp_Mass_Norm_1),
       col = 'red')

# Number of samples cooked in replicate so far
replicated_data = data %>%
  group_by(Sample_ID) %>% 
  filter(n() > 1) %>%
  select(Sample_ID, Rep, Pericarp_Mass_Avg_Norm) %>%
  pivot_wider(names_from = Rep, values_from = Pericarp_Mass_Avg_Norm)

# Plot the data
plot(replicated_data$A,
     replicated_data$B,
     xlab = if(trans_needed) 'log(Replicate A)' else 'Replicate A',
     ylab = if(trans_needed) 'log(Replicate B)' else 'Replicate B',
     main = 'Replicate Correlation')
abline(lm(replicated_data$B ~ replicated_data$A),
       col = 'red')

# Summary Table:
summary_table = tibble(Sample_Type = c('Subsample',
                                       'Replicate'),
                       N = c(nrow(data),
                             nrow(replicated_data)),
                       Spearman_R = c(signif(cor(data$Dry_Pericarp_Mass_Norm_1,
                                                 data$Dry_Pericarp_Mass_Norm_2,
                                                 method = 'spearman',
                                                 use = 'complete.obs'), 3),
                                      signif(cor(replicated_data$A,
                                                 replicated_data$B,
                                                 method = 'spearman',
                                                 use = 'complete.obs'), 3)),
                       Pearson_R = c(signif(cor(data$Dry_Pericarp_Mass_Norm_1,
                                                data$Dry_Pericarp_Mass_Norm_2,
                                                use = 'complete.obs'), 3),
                                     signif(cor(replicated_data$A,
                                                replicated_data$B,
                                                use = 'complete.obs'), 3)),
                       T_Test_P_Value = c(signif(t.test(data$Dry_Pericarp_Mass_Norm_1,
                                                        data$Dry_Pericarp_Mass_Norm_2,
                                                        paired = TRUE)$p.value, 3),
                                          signif(t.test(replicated_data$A,
                                                        replicated_data$B,
                                                        paired = TRUE)$p.value, 3)),
                       Log_Transformed = trans_needed)

# Create a mixed linear model to determine the proportion of variance explained by variance components
data_for_model = data

# Check model assumptions of a and b hotplates being similar
model = lm(A ~ B, data = replicated_data)
opar = par(mfrow = c(2,2))
print(plot(model, which = 1:4))
dev.off()
# Sample ID is a combination of genotype and year information that is not broken out,
# Cook Date is to determine if there is a strong date effect that could explain the lack of correlation between replicates while subsample correlation is decent
# Cook time, steep time, and pH are to determine what proportion of variance the cooking parameters have on the pericarp mass
# Peeler is to ensure that we are not seeing a strong difference between the two people peeling samples.
model = lmer(Pericarp_Mass_Avg_Norm ~ (1|Sample_ID) + (1|Rep) + (1|Peeler), data = data_for_model)

# Summarise Findings:
if(trans_needed){
  paste('Log transformed Pericarp Mass is not normally distributed so a log transformation was applied')
}

summary(model)$varcor %>%
  as_tibble() %>%
  mutate(Total_SS = sum(vcov),
         PVE = vcov / Total_SS) %>%
  select(grp, PVE) %>%
  rename(Variable = grp)

print(summary_table)