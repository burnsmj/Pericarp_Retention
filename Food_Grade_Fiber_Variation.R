### Finding Fiber Variation in Food Grade Lines
### Michael Burns
### 4/21/22

# Libraries
library(tidyverse)
library(readxl)

# Data
raw_scans = read_xlsx('/Users/michael/Downloads/corn h.xlsx')

# Filter for food grade
clean_scans = raw_scans %>%
  filter(str_detect(Id, 'IL_[A-Z]') | str_detect(Id, 'NE_[A-Z]')) %>%
  select(Id, `Fiber As is`) %>%
  mutate(Rep = str_extract(Id, '[1-2]$'),
         Id = str_remove(Id, '_[1-2]$')) %>%
  rename(Sample_ID = Id,
         Fiber = `Fiber As is`) %>%
  filter(!str_detect(Sample_ID, '[0-9]')) %>%
  pivot_wider(id_cols = Sample_ID,
              values_from = Fiber,
              names_from = Rep,
              names_prefix = 'Rep_') %>%
  mutate(Diff = Rep_1 - Rep_2,
         Fiber = (Rep_1 + Rep_2) / 2) %>%
  arrange(desc(Fiber))

Pep_2020_SampleIDs = c('AM', 'S', 'J', 'C', 'AK', 'AD', 'V', 'AC',
                       'Z', 'L', 'P', 'D', 'Y', 'B', 'A', 'AH', 'AA',
                       'W', 'Q', 'E', 'O', 'K', 'N', 'M', 'U', 'R',
                       'AP', 'H', 'F', 'AJ', 'AL', 'X', 'AN', 'AG', 'T',
                       'AI', 'AO')

view(clean_scans)

# Filter for IL and NE
IL_Scans = clean_scans %>%
  mutate(State = str_extract(Sample_ID, '^[A-Z][A-Z]'),
         Sample = str_extract(Sample_ID, '_.*'),
         Sample = str_remove(Sample, '_')) %>%
  filter(State == 'IL',
         Sample %in% Pep_2020_SampleIDs)

NE_Scans = clean_scans %>%
  mutate(State = str_extract(Sample_ID, '^[A-Z][A-Z]'),
         Sample = str_extract(Sample_ID, '_.*'),
         Sample = str_remove(Sample, '_')) %>%
  filter(State == 'NE',
         Sample %in% Pep_2020_SampleIDs)

# Sample 5 from each state
IL_Scans[seq(1,nrow(IL_Scans), round(nrow(IL_Scans) / 5)),] # Take out a middle sample since this gives 6.  I am taking out IL_Y due to it's proximity to other samples.

NE_Scans[seq(1,nrow(NE_Scans), round(nrow(NE_Scans) / 5), ),] # Take out a middle sample since this gives 6.  I am taking out NE_D due to it's proximity to other samples.
