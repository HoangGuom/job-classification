# Model Card — Job Level Classifier

## Model overview

| Thuộc tính | Giá trị |
|---|---|
| Bài toán | Phân loại đa lớp |
| Mục tiêu | Dự đoán cấp bậc nghề nghiệp từ tin tuyển dụng |
| Model cuối | Linear SVM (`LinearSVC`) |
| Dữ liệu sau làm sạch | 8.052 mẫu |
| Training set | 6.441 mẫu |
| Test set | 1.611 mẫu |
| Số lớp | 6 |
| Metric lựa chọn | Macro-F1 |

## Input

- Job title
- Location
- Job description
- Function
- Industry

## Output

Model dự đoán một trong sáu nhãn:

1. `specialist`
2. `senior_specialist_or_project_manager`
3. `manager_team_leader`
4. `bereichsleiter`
5. `director_business_unit_leader`
6. `managing_director_small_medium_company`

## Pipeline

```text
Input
  ├── Title + Description + Industry
  │     └── TF-IDF unigram/bigram
  ├── Location
  │     └── One-Hot Encoding
  └── Function
        └── One-Hot Encoding
             ↓
     Linear SVM with balanced class weights
             ↓
       Predicted career level
```

## Kết quả

| Model | Accuracy | Macro-F1 | Weighted-F1 | Balanced accuracy |
|---|---:|---:|---:|---:|
| Baseline | 0,5369 | 0,1165 | 0,3752 | 0,1667 |
| Logistic Regression | 0,7716 | 0,5286 | 0,7702 | **0,5357** |
| **Linear SVM** | **0,7871** | **0,5406** | **0,7793** | 0,4917 |

Linear SVM được chọn vì đạt macro-F1 cao nhất.

## Cách kiểm tra file `.joblib`

File `.joblib` là file nhị phân chứa pipeline đã huấn luyện, không thể đọc trực tiếp như file code.

Chạy:

```powershell
python inspect_model.py
```

Để dùng model dự đoán qua giao diện:

```powershell
python -m streamlit run app.py
```

## Hạn chế

- Dữ liệu mất cân bằng rất mạnh.
- Lớp nhỏ nhất chỉ có 4 mẫu.
- Relative score từ Linear SVM không phải xác suất đã hiệu chỉnh.
- Model chỉ nên được xem là bản demo kỹ thuật, không dùng trực tiếp để ra quyết định nhân sự.
