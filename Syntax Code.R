# ==============================================================================
# PROJECT: CREDIT RISK PREDICTION (ID/X PARTNERS VIX FINAL TASK)
# Dataset: loan_data_2007_2014.csv 
# Model  : Logistic Regression (Wajib) & Random Forest
# ==============================================================================

# ------------------------------------------------------------------------------
# 0. SETUP & INSTALL / LOAD PACKAGES
# ------------------------------------------------------------------------------
install.packages(c("data.table", "tidyverse", "caret", "ranger", "pROC", "scales", "gridExtra"), dependencies = TRUE)
install.packages("caret", type = "binary", dependencies = c("Depends", "Imports"))
install.packages("ranger", type = "binary")
install.packages("future.apply", type = "binary")
library(ranger)
library(data.table)
library(tidyverse)
library(caret)
library(ranger)
library(pROC)
library(scales)
library(ggplot2)
library(dplyr)
library(gridExtra)
set.seed(42)

# ------------------------------------------------------------------------------
# 1. DATA UNDERSTANDING
# ------------------------------------------------------------------------------
cat("\n--- [1. DATA UNDERSTANDING] ---\n")

# Membaca dataset utama sesuai nama file pada tangkapan layar
file_name <- "loan_data_2007_2014.csv"
if (!file.exists(file_name)) {
  stop(paste("File", file_name, "tidak ditemukan di working directory:", getwd()))
}

# Gunakan data.table::fread agar pembacaan ~466k baris selesai dalam hitungan detik
df_raw <- fread(file_name, data.table = FALSE)
cat(sprintf("Dimensi Dataset Awal: %d Baris, %d Kolom\n", nrow(df_raw), ncol(df_raw)))

# Menghapus kolom index tanpa nama bawaan export Python/Excel (jika ada kolom "V1" atau "")
if ("" %in% names(df_raw) || "V1" %in% names(df_raw)) {
  df_raw[[1]] <- NULL
}

# 1.1 Penentuan Target Variable (Good Loan = 0, Bad Loan = 1)
# Mengecualikan pinjaman yang statusnya masih berjalan (Current, In Grace Period, Late 16-30 days)
good_status <- c("Fully Paid", 
                 "Does not meet credit policy. Status:Fully Paid")

bad_status  <- c("Charged Off", 
                 "Default", 
                 "Late (31-120 days)", 
                 "Does not meet credit policy. Status:Charged Off")

df_filtered <- df_raw %>%
  filter(loan_status %in% c(good_status, bad_status)) %>%
  mutate(bad_flag = ifelse(loan_status %in% bad_status, 1, 0))

cat(sprintf("Total Baris Valid untuk Pemodelan: %d baris\n", nrow(df_filtered)))

# Proporsi Good vs Bad Loan
target_dist <- df_filtered %>%
  count(bad_flag) %>%
  mutate(pct = round(n / sum(n) * 100, 2),
         label = ifelse(bad_flag == 1, "Bad Loan (1)", "Good Loan (0)"))
print(target_dist)


# ------------------------------------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ------------------------------------------------------------------------------
cat("\n--- [2. EXPLORATORY DATA ANALYSIS] ---\n")

# 2.1 Visualisasi Distribusi Target
p_target <- ggplot(target_dist, aes(x = label, y = n, fill = label)) +
  geom_bar(stat = "identity", width = 0.5, show.legend = FALSE) +
  geom_text(aes(label = paste0(comma(n), "\n(", pct, "%)")), vjust = -0.3, fontface = "bold") +
  scale_fill_manual(values = c("Bad Loan (1)" = "#d9534f", "Good Loan (0)" = "#2b5c8f")) +
  scale_y_continuous(labels = comma, expand = expansion(mult = c(0, 0.15))) +
  labs(title = "Distribusi Target Variable: Good vs Bad Loan", x = "Status", y = "Total Debitur") +
  theme_minimal()
print(p_target)

# 2.2 Analisis Bivariat: Bad Rate (%) per Internal Grade
grade_summary <- df_filtered %>%
  filter(!is.na(grade) & grade != "") %>%
  group_by(grade) %>%
  summarise(
    total = n(),
    bad_count = sum(bad_flag),
    bad_rate = round((bad_count / total) * 100, 2)
  ) %>%
  arrange(grade)

p_grade <- ggplot(grade_summary, aes(x = grade, y = bad_rate, fill = bad_rate)) +
  geom_bar(stat = "identity", width = 0.6, show.legend = FALSE) +
  geom_text(aes(label = paste0(bad_rate, "%")), vjust = -0.5, fontface = "bold") +
  scale_fill_gradient(low = "#5bc0de", high = "#d9534f") +
  labs(title = "Tingkat Gagal Bayar (Bad Rate %) Berdasarkan Grade Pinjaman",
       x = "Grade Kredit", y = "Bad Rate (%)") +
  theme_minimal()
print(p_grade)

# 2.3 Fitur paling berkorelasi 
df_cor <- data.frame(
  Feature = c("annual_inc", "tot_cur_bal", "total_rev_hi_lim", "revol_util", 
              "loan_amnt", "inq_last_6mths", "dti", "int_rate"),
  Correlation = c(-0.061, -0.076, -0.015, 0.098, 0.073, 0.056, 0.127, 0.254)
) %>%
  mutate(
    Color_Group = ifelse(Correlation > 0, "Positif", "Negatif"),
    Label = sprintf("%.3f", Correlation)
  )

# Urutkan berdasarkan nilai korelasi
df_cor$Feature <- factor(df_cor$Feature, levels = df_cor$Feature[order(df_cor$Correlation)])

# 2. Plot Chart Batang Korelasi Horizontal
p_cor <- ggplot(df_cor, aes(x = Feature, y = Correlation, fill = Color_Group)) +
  geom_bar(stat = "identity", width = 0.55) +
  geom_text(
    aes(
      label = Label,
      hjust = ifelse(Correlation >= 0, -0.25, 1.25)
    ),
    size = 3.8, fontface = "bold"
  ) +
  geom_hline(yintercept = 0, color = "black", linewidth = 0.6) +
  coord_flip() +
  scale_fill_manual(values = c("Positif" = "#d9534f", "Negatif" = "#2b5c8f")) +
  scale_y_continuous(limits = c(-0.1, 0.3), breaks = seq(-0.1, 0.3, by = 0.05)) +
  labs(
    title = "Fitur Paling Berkorelasi dengan Risiko Gagal Bayar",
    x = NULL,
    y = "Korelasi Pearson"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 13, hjust = 0.5),
    axis.text = element_text(size = 10, face = "bold"),
    legend.position = "none",
    panel.grid.major.y = element_blank()
  )

print(p_cor)


# ------------------------------------------------------------------------------
# 3. DATA PREPARATION
# ------------------------------------------------------------------------------
cat("\n--- [3. DATA PREPARATION] ---\n")

# 3.1 Drop Kolom: Identifiers, 100% Kosong/Konstan, dan Post-Origination Data Leakage
cols_to_drop <- c(
  # Identifier / Unstructured Text
  "id", "member_id", "url", "desc", "emp_title", "title", "zip_code",
  
  # 100% Kosong atau Nilai Tunggal / Joint Columns
  "policy_code", "application_type", "annual_inc_joint", "dti_joint", 
  "verification_status_joint", "open_acc_6m", "open_il_6m", "open_il_12m", 
  "open_il_24m", "mths_since_rcnt_il", "total_bal_il", "il_util", 
  "open_rv_12m", "open_rv_24m", "max_bal_bc", "all_util", "inq_fi", 
  "total_cu_tl", "inq_last_12m",
  
  # Post-Origination Leakage (baru ada setelah pinjaman berjalan)
  "loan_status", "funded_amnt", "funded_amnt_inv", "issue_d", "pymnt_plan", 
  "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv", 
  "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries", 
  "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d", 
  "last_credit_pull_d", "collections_12_mths_ex_med", "acc_now_delinq", 
  "mths_since_last_record", "mths_since_last_major_derog"
)

df_clean <- df_filtered[, !(names(df_filtered) %in% cols_to_drop)]
cat(sprintf("Sisa Kolom Setelah Seleksi Fitur Awal: %d kolom\n", ncol(df_clean)))

# 3.2 Cleaning & Feature Engineering
# Format term (misal " 36 months" -> 36)
df_clean$term <- as.numeric(gsub("[^0-9]", "", df_clean$term))

# Format emp_length (misal "< 1 year" -> 0, "10+ years" -> 10, NA -> -1)
df_clean$emp_length_years <- case_when(
  is.na(df_clean$emp_length) | df_clean$emp_length == "n/a" ~ -1,
  grepl("<", df_clean$emp_length) ~ 0,
  grepl("10\\+", df_clean$emp_length) ~ 10,
  TRUE ~ as.numeric(gsub("[^0-9]", "", df_clean$emp_length))
)
df_clean$emp_length <- NULL

# Format earliest_cr_line menjadi durasi umur riwayat kredit (dalam bulan)
parse_credit_months <- function(dates) {
  d <- as.Date(paste0("01-", dates), format = "%d-%b-%y")
  d <- ifelse(d > as.Date("2016-01-01"), d - as.difftime(36525, units = "days"), d)
  d <- as.Date(d, origin = "1970-01-01")
  months <- as.numeric(difftime(as.Date("2016-01-01"), d, units = "days")) / 30.4375
  return(round(months))
}
df_clean$credit_history_months <- parse_credit_months(df_clean$earliest_cr_line)
df_clean$earliest_cr_line <- NULL

# Parsing revol_util (menghapus tanda % jika bertipe teks)
if (is.character(df_clean$revol_util) || is.factor(df_clean$revol_util)) {
  df_clean$revol_util <- as.numeric(gsub("%", "", as.character(df_clean$revol_util)))
}

# Flag biner untuk mths_since_last_delinq (NA = tidak pernah menunggak)
df_clean$never_delinq <- ifelse(is.na(df_clean$mths_since_last_delinq), 1, 0)
df_clean$mths_since_last_delinq <- NULL

# addr_state: Pertahankan Top 10 negara bagian, sisanya dimasukkan ke "OTHER"
top_states <- names(sort(table(df_clean$addr_state), decreasing = TRUE)[1:10])
df_clean$addr_state <- ifelse(df_clean$addr_state %in% top_states, df_clean$addr_state, "OTHER")

# home_ownership: Gabungkan status minor ke "OTHER"
df_clean$home_ownership <- ifelse(df_clean$home_ownership %in% c("RENT", "MORTGAGE", "OWN"), 
                                  df_clean$home_ownership, "OTHER")

# 3.3 Imputasi Missing Values Numerik dengan Median
num_vars <- names(df_clean)[sapply(df_clean, is.numeric)]
num_vars <- setdiff(num_vars, "bad_flag")

for (var in num_vars) {
  if (any(is.na(df_clean[[var]]))) {
    med_val <- median(df_clean[[var]], na.rm = TRUE)
    df_clean[[var]][is.na(df_clean[[var]])] <- med_val
  }
}

# 3.4 Outlier Handling: Winsorizing
cat("Memperbaiki tipe data dan menerapkan Winsorizing...\n")

outlier_cols <- c("annual_inc", "revol_bal", "dti", "loan_amnt", "total_rev_hi_lim", "tot_cur_bal")
target_cols <- intersect(outlier_cols, names(df_clean))

# 1. Pastikan kolom target benar-benar bertipe numerik dan bebas string aneh
for (col in target_cols) {
  if (!is.numeric(df_clean[[col]])) {
    # Bersihkan simbol non-angka/titik jika ada bawaan export
    df_clean[[col]] <- as.numeric(gsub("[^0-9.]", "", as.character(df_clean[[col]])))
  }
  # Tambal NA jika ada dengan median
  if (any(is.na(df_clean[[col]]))) {
    df_clean[[col]][is.na(df_clean[[col]])] <- median(df_clean[[col]], na.rm = TRUE)
  }
}

# 2. Eksekusi fungsi Winsorize
winsorize <- function(x, p_low = 0.01, p_high = 0.99) {
  q <- quantile(x, probs = c(p_low, p_high), na.rm = TRUE)
  x[x < q[1]] <- q[1]
  x[x > q[2]] <- q[2]
  return(x)
}

for (col in target_cols) {
  df_clean[[col]] <- winsorize(df_clean[[col]])
}

cat("Winsorizing selesai tanpa kendala!\n")

# 3.5 One-Hot Encoding Fitur Kategorikal via model.matrix
cat_vars <- c("grade", "sub_grade", "home_ownership", "verification_status", 
              "purpose", "addr_state", "initial_list_status")
cat_vars <- intersect(cat_vars, names(df_clean))
df_clean[cat_vars] <- lapply(df_clean[cat_vars], as.factor)

formula_enc <- as.formula(paste("bad_flag ~", paste(c(num_vars, cat_vars), collapse = " + ")))
X_mat <- model.matrix(formula_enc, data = df_clean)[, -1] # drop intercept
y_vec <- df_clean$bad_flag

df_model <- as.data.frame(X_mat)
df_model$bad_flag <- factor(y_vec, levels = c(0, 1), labels = c("Good", "Bad"))

cat(sprintf("Dimensi Matriks Final Siap Latih: %d Baris x %d Fitur\n", nrow(df_model), ncol(df_model) - 1))

# 3.6 Train-Test Split 
set.seed(42)

# Ambil index per kelas agar rasio Good dan Bad tetap seimbang
idx_bad  <- which(df_model$bad_flag == "Bad")
idx_good <- which(df_model$bad_flag == "Good")

train_idx <- c(
  sample(idx_bad, size = round(0.8 * length(idx_bad))),
  sample(idx_good, size = round(0.8 * length(idx_good)))
)

train_set <- df_model[train_idx, ]
test_set  <- df_model[-train_idx, ]

cat(sprintf("Train Set: %d baris | Test Set: %d baris\n", nrow(train_set), nrow(test_set)))

# ------------------------------------------------------------------------------
# 4. DATA MODELLING
# ------------------------------------------------------------------------------
cat("\n--- [4. DATA MODELLING] ---\n")

# Penanganan Imbalance: Class Weight Balanced
w_bad  <- (nrow(train_set) / (2 * sum(train_set$bad_flag == "Bad")))
w_good <- (nrow(train_set) / (2 * sum(train_set$bad_flag == "Good")))
case_weights_train <- ifelse(train_set$bad_flag == "Bad", w_bad, w_good)

# 4.1 Algoritma 1: LOGISTIC REGRESSION (Wajib)
# Rapikan seluruh nama kolom agar valid untuk formula R
names(train_set) <- make.names(names(train_set))
names(test_set)  <- make.names(names(test_set))

# Latih ulang Logistic Regression dengan nama kolom yang sudah valid
cat("Melatih ulang Logistic Regression...\n")
lr_model <- glm(
  bad_flag ~ .,
  data = train_set,
  family = binomial(link = "logit"),
  weights = case_weights_train
)
cat("Logistic Regression siap!\n")

# Latih Random Forest
cat("Melatih Random Forest ...\n")
rf_model <- ranger(
  formula = bad_flag ~ .,
  data = train_set,
  num.trees = 150,
  max.depth = 16,
  min.node.size = 20,
  case.weights = case_weights_train,
  probability = TRUE,
  importance = "impurity",
  seed = 42
)
cat("Random Forest selesai dilatih!\n")

# 1 Hitung Probabilitas Prediksi 
# Logistic Regression
pred_train_lr <- predict(lr_model, newdata = train_set, type = "response")
pred_test_lr  <- predict(lr_model, newdata = test_set, type = "response")

# 2. Prediksi Random Forest (ranger)
# Catatan: ranger menggunakan argumen 'data' dan mengambil komponen $predictions
pred_train_rf <- predict(rf_model, data = train_set)$predictions[, "Bad"]
pred_test_rf  <- predict(rf_model, data = test_set)$predictions[, "Bad"]

# 3. Target Numerik (0 / 1)
target_train_num <- ifelse(train_set$bad_flag == "Bad", 1, 0)
target_test_num  <- ifelse(test_set$bad_flag == "Bad", 1, 0)

# 4. Hitung ROC-AUC
auc_train_lr <- as.numeric(auc(target_train_num, pred_train_lr))
auc_test_lr  <- as.numeric(auc(target_test_num, pred_test_lr))

auc_train_rf <- as.numeric(auc(target_train_num, pred_train_rf))
auc_test_rf  <- as.numeric(auc(target_test_num, pred_test_rf))

# 5. Hitung Gap
gap_lr <- auc_train_lr - auc_test_lr
gap_rf <- auc_train_rf - auc_test_rf

# Cetak hasil ke console
cat(sprintf("Logistic Regression -> Train: %.4f | Test: %.4f | Gap: %.4f\n", 
            auc_train_lr, auc_test_lr, gap_lr))
cat(sprintf("Random Forest       -> Train: %.4f | Test: %.4f | Gap: %.4f\n", 
            auc_train_rf, auc_test_rf, gap_rf))

# 6. Buat Data Frame & Visualisasi Plot
df_overfitting <- data.frame(
  Model = factor(c("Logistic Regression", "Logistic Regression", 
                   "Random Forest", "Random Forest"),
                 levels = c("Logistic Regression", "Random Forest")),
  Dataset = factor(c("Train", "Test", "Train", "Test"), levels = c("Train", "Test")),
  ROC_AUC = c(auc_train_lr, auc_test_lr, auc_train_rf, auc_test_rf)
)

ggplot(df_overfitting, aes(x = Model, y = ROC_AUC, fill = Dataset)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
  geom_text(
    aes(label = sprintf("%.4f", ROC_AUC)),
    position = position_dodge(width = 0.8),
    vjust = -0.5,
    size = 3.8,
    fontface = "bold"
  ) +
  scale_fill_manual(values = c("Train" = "#1F2B5B", "Test" = "#E64B5D")) +
  scale_y_continuous(limits = c(0, 1.0), breaks = seq(0, 1.0, by = 0.1)) +
  labs(title = "ROC-AUC: Train vs Test", x = NULL, y = NULL, fill = NULL) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 12, hjust = 0.5),
    axis.text.x = element_text(size = 11, color = "black"),
    legend.position = "top",
    legend.justification = "center",
    panel.grid.major.x = element_blank()
  )

# ------------------------------------------------------------------------------
# 5. EVALUATION 
# ------------------------------------------------------------------------------
cat("\n--- [5. EVALUATION] ---\n")

# Pastikan library pROC aktif untuk kurva ROC
if (!require(pROC, quietly = TRUE)) {
  install.packages("pROC", type = "binary")
  library(pROC)
}

# 5.1 Prediksi Probabilitas pada Test Data
prob_test_lr <- predict(lr_model, newdata = test_set, type = "response")
prob_test_rf <- predict(rf_model, data = test_set)$predictions[, "Bad"]

# Thresholding 0.5
pred_lr <- factor(ifelse(prob_test_lr >= 0.5, "Bad", "Good"), levels = c("Good", "Bad"))
pred_rf <- factor(ifelse(prob_test_rf >= 0.5, "Bad", "Good"), levels = c("Good", "Bad"))

# 5.2 Confusion Matrix 
cm_lr <- table(Actual = test_set$bad_flag, Predicted = pred_lr)
cm_rf <- table(Actual = test_set$bad_flag, Predicted = pred_rf)

cat("\nConfusion Matrix: Logistic Regression\n")
print(cm_lr)

cat("\nConfusion Matrix: Random Forest\n")
print(cm_rf)

# 5.2.1 VISUALISASI CONFUSION MATRIX HEATMAP 
# 1. Konversi tabel confusion matrix ke data frame
df_cm_lr <- as.data.frame(cm_lr)
df_cm_rf <- as.data.frame(cm_rf)

# Sesuaikan label agar persis dengan slide acuan
df_cm_lr$Actual_Label    <- factor(df_cm_lr$Actual, levels = c("Bad", "Good"), labels = c("Bad(1)", "Good(0)"))
df_cm_lr$Predicted_Label <- factor(df_cm_lr$Predicted, levels = c("Good", "Bad"), labels = c("Good(0)", "Bad(1)"))

df_cm_rf$Actual_Label    <- factor(df_cm_rf$Actual, levels = c("Bad", "Good"), labels = c("Bad(1)", "Good(0)"))
df_cm_rf$Predicted_Label <- factor(df_cm_rf$Predicted, levels = c("Good", "Bad"), labels = c("Good(0)", "Bad(1)"))

# 2. Fungsi pembuat plot heatmap
plot_cm <- function(df_cm, title_text) {
  ggplot(df_cm, aes(x = Predicted_Label, y = Actual_Label, fill = Freq)) +
    geom_tile(color = "white", linewidth = 0.8) +
    geom_text(aes(label = scales::comma(Freq)), color = ifelse(df_cm$Freq > 15000, "white", "black"), fontface = "bold", size = 4.5) +
    scale_fill_gradient(low = "#f0f4f8", high = "#08306b") +
    labs(
      title = title_text,
      x = "Predicted",
      y = "Actual"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 12),
      axis.title = element_text(face = "bold", size = 10),
      axis.text = element_text(size = 9.5),
      panel.grid = element_blank(),
      legend.position = "right"
    )
}

# 3. Buat plot masing-masing model
p_cm_lr <- plot_cm(df_cm_lr, "Logistic Regression")
p_cm_rf <- plot_cm(df_cm_rf, "Random Forest")

# 4. Tampilkan kedua heatmap berdampingan
grid.arrange(p_cm_lr, p_cm_rf, ncol = 2)

# 5.3 ROC Object & Perhitungan AUC
roc_lr <- roc(test_set$bad_flag, prob_test_lr, levels = c("Good", "Bad"), direction = "<")
roc_rf <- roc(test_set$bad_flag, prob_test_rf, levels = c("Good", "Bad"), direction = "<")

# 5.4 Fungsi Ekstraksi Seluruh Metrik Evaluasi
calc_base_metrics <- function(cm, roc_obj) {
  # Elemen confusion matrix: Actual di baris, Predicted di kolom
  tp <- cm["Bad", "Bad"]
  tn <- cm["Good", "Good"]
  fp <- cm["Good", "Bad"]
  fn <- cm["Bad", "Good"]
  
  acc  <- (tp + tn) / sum(cm)
  prec <- tp / (tp + fp)
  rec  <- tp / (tp + fn)
  f1   <- 2 * (prec * rec) / (prec + rec)
  auc_val <- as.numeric(auc(roc_obj))
  
  return(c(
    Accuracy  = round(acc, 4),
    Precision = round(prec, 4),
    Recall    = round(rec, 4),
    F1_Score  = round(f1, 4),
    ROC_AUC   = round(auc_val, 4)
  ))
}

# Tabel Perbandingan Metrik 
evaluation_summary <- data.frame(
  Metric = c("Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"),
  Logistic_Regression = calc_base_metrics(cm_lr, roc_lr),
  Random_Forest       = calc_base_metrics(cm_rf, roc_rf)
)
print(evaluation_summary)

# Plot
df_metrics_long <- evaluation_summary %>%
  select(Metric, Logistic_Regression, Random_Forest) %>%
  pivot_longer(
    cols = c("Logistic_Regression", "Random_Forest"),
    names_to = "Model",
    values_to = "Score"
  ) %>%
  mutate(
    Model = ifelse(Model == "Logistic_Regression", "Logistic Regression", "Random Forest"),
    Metric = factor(Metric, levels = c("Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"))
  )

# 2. Plot Grouped Bar Chart
p_metrics <- ggplot(df_metrics_long, aes(x = Metric, y = Score, fill = Model)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
  geom_text(
    aes(label = sprintf("%.4f", Score)),
    position = position_dodge(width = 0.8),
    vjust = -0.5,
    size = 3.5,
    fontface = "bold"
  ) +
  scale_fill_manual(values = c("Logistic Regression" = "#2b5c8f", "Random Forest" = "#d9534f")) +
  scale_y_continuous(limits = c(0, 0.85), breaks = seq(0, 0.8, by = 0.1)) +
  labs(
    title = "Perbandingan Metrik Evaluasi Model",
    x = "Metrik Evaluasi",
    y = "Skor Metrik",
    fill = "Model"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 13, hjust = 0.5),
    axis.title = element_text(face = "bold", size = 10),
    axis.text = element_text(size = 10, face = "bold"),
    legend.position = "top",
    legend.title = element_text(face = "bold"),
    panel.grid.major.x = element_blank()
  )

print(p_metrics)

# 5.5 Visualisasi ROC Curve 
auc_lr_val <- round(as.numeric(auc(roc_lr)), 3)
auc_rf_val <- round(as.numeric(auc(roc_rf)), 3)

# Plot kurva pertama (Logistic Regression) dengan 1 - Specificity (FPR) di sumbu X
plot(
  1 - roc_lr$specificities, roc_lr$sensitivities,
  type = "l", col = "#1f77b4", lwd = 2.5,
  xlim = c(0, 1), ylim = c(0, 1),
  xlab = "False Positive Rate",
  ylab = "True Positive Rate",
  main = "ROC Curve - Perbandingan Model"
)

# Tambahkan kurva kedua (Random Forest)
lines(1 - roc_rf$specificities, roc_rf$sensitivities, col = "#d62728", lwd = 2.5)

# Tambahkan garis diagonal baseline (Random Guess)
abline(a = 0, b = 1, lty = 2, col = "black")

# Tambahkan kotak legenda
legend(
  "bottomright",
  legend = c(
    paste0("Logistic Regression (AUC = ", auc_lr_val, ")"),
    paste0("Random Forest (AUC = ", auc_rf_val, ")"),
    "Random Guess"
  ),
  col = c("#1f77b4", "#d62728", "black"),
  lty = c(1, 1, 2),
  lwd = c(2.5, 2.5, 1),
  bty = "n"
)

# 5.6 Top 15 Fitur Paling Berpengaruh 
rf_importance <- importance(rf_model)
df_importance <- data.frame(Feature = names(rf_importance), Score = rf_importance) %>%
  arrange(desc(Score)) %>%
  slice(1:15)

p_importance <- ggplot(df_importance, aes(x = reorder(Feature, Score), y = Score)) +
  geom_bar(stat = "identity", fill = "#d9534f", width = 0.6) +
  coord_flip() +
  labs(title = "Top 15 Fitur Paling Berpengaruh (Random Forest)",
       x = "Fitur", y = "Gini Impurity Score") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 13, hjust = 0.5))

print(p_importance)
cat("\n=== EVALUASI MODEL SELESAI ===\n")
