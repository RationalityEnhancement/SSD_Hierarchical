library(ez)
library(nparLD)


# Read data
dat = read.csv("./main_excluded.csv")
df <- dat[,c("Participant", "Condition", "TrialId", "ExpectedScore", "NumClicks", "Score")]

df$Participant <- as.factor(df$Participant)
df$TrialId <- as.factor(df$TrialId)
df$Condition <- as.factor(df$Condition)

runAnova <- function(df){
  # This one doesn't work for some conditions
  y <- df$ExpectedScore
  time <- df$TrialId
  group <- df$Condition
  subject <- df$Participant
  f1 <- f1.ld.f1(y, time, group, subject, time.name = "Trial", group.name = "Condition", description = FALSE, order.warning=FALSE, show.covariance = TRUE) #
  print("F1 Walt test")
  print(f1$Wald.test)
  print("F1 Anova test")
  print(f1$ANOVA.test)
  print("Anova box")
  print(f1$ANOVA.test.mod.Box)
  print("F1 pairwise")
  print(f1$pair.comparison)
}

runAnova(df)

