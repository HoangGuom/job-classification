# Job Level Classification

Dự án Machine Learning phân loại **6 cấp bậc nghề nghiệp** từ nội dung tin tuyển dụng.

Toàn bộ quy trình được trình bày theo từng bước trong notebook:

[`notebooks/job_level_classification.ipynb`](notebooks/job_level_classification.ipynb)

## Nội dung dự án

- Làm sạch và phân tích hơn 8.000 tin tuyển dụng.
- Biểu diễn văn bản bằng TF-IDF.
- Mã hóa location và function bằng One-Hot Encoding.
- Xử lý dữ liệu mất cân bằng bằng class weighting.
- So sánh baseline, Logistic Regression và Linear SVM.
- Đánh giá bằng accuracy, Macro-F1 và confusion matrix.
- Nhập một tin tuyển dụng mới và dự đoán cấp bậc.

## Kết quả

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Baseline | 53,69% | 11,65% |
| Logistic Regression | 77,16% | 52,86% |
| **Linear SVM** | **78,71%** | **54,06%** |

## Chạy notebook

```bash
python -m venv .venv
pip install -r requirements.txt
jupyter notebook notebooks/job_level_classification.ipynb
```

Đặt file dữ liệu `final_project.ods` tại thư mục gốc trước khi chạy toàn bộ notebook.

Dataset không được commit do chưa xác nhận quyền phân phối.
