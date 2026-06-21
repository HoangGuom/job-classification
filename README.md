# Job Level Classification

Portfolio Machine Learning project dự đoán **cấp bậc nghề nghiệp** từ nội dung tin tuyển dụng.

Mục tiêu của dự án không chỉ là đạt accuracy cao, mà còn thể hiện một quy trình ML hoàn chỉnh: phân tích dữ liệu, xử lý mất cân bằng, xây dựng baseline, thử nghiệm mô hình, lựa chọn metric phù hợp, đóng gói pipeline, kiểm thử và triển khai giao diện dự đoán.

## Bài toán dự đoán

### Input

- `title`: tiêu đề công việc
- `location`: địa điểm
- `description`: mô tả công việc
- `function`: nhóm chức năng
- `industry`: ngành nghề

### Output

Một trong sáu cấp bậc:

1. Specialist
2. Senior Specialist / Project Manager
3. Manager / Team Leader
4. Department Head
5. Director / Business Unit Leader
6. Managing Director (SME)

Ứng dụng trả thêm top 3 kết quả cùng relative score. Với Linear SVM, score này dùng để xếp hạng và **không phải xác suất đã hiệu chỉnh**.

## Kết quả

Dữ liệu sau làm sạch có **8.052 tin tuyển dụng**. Tập test chiếm 20% và được chia theo stratified split.

| Model | Accuracy | Macro F1 | Weighted F1 | Balanced accuracy |
|---|---:|---:|---:|---:|
| Most-frequent baseline | 0.5369 | 0.1165 | 0.3752 | 0.1667 |
| Logistic Regression | 0.7716 | 0.5286 | 0.7702 | **0.5357** |
| **Linear SVM** | **0.7871** | **0.5406** | **0.7793** | 0.4917 |

Mô hình cuối được chọn theo **macro-F1**, vì dữ liệu mất cân bằng rất mạnh. Accuracy một mình có thể che giấu việc mô hình bỏ qua các lớp hiếm.

![Normalized confusion matrix](artifacts/confusion_matrix.png)

## Mô hình đã được cải thiện như thế nào?

1. **Baseline:** luôn dự đoán lớp phổ biến nhất, accuracy 53,69% nhưng macro-F1 chỉ 11,65%.
2. **Feature engineering:** ghép title, description và industry; title được lặp lại để tăng trọng số tín hiệu chức danh.
3. **Text representation:** TF-IDF unigram + bigram, loại stop words, sublinear term frequency và giới hạn 50.000 đặc trưng.
4. **Categorical features:** One-Hot Encoding cho location và function, tự xử lý category chưa gặp.
5. **Class imbalance:** dùng `class_weight="balanced"` thay vì chỉ tối ưu lớp chiếm đa số.
6. **Model comparison:** so sánh Logistic Regression và Linear SVM trên cùng train/test split.
7. **Model selection:** chọn Linear SVM theo macro-F1, tăng từ 0,1165 lên 0,5406 so với baseline.
8. **Productionization:** đóng gói toàn bộ preprocessing + model trong một scikit-learn Pipeline để training và inference dùng đúng một quy trình.

## Công nghệ

- Python 3.12
- pandas, NumPy
- scikit-learn
- TF-IDF, One-Hot Encoding
- Logistic Regression, Linear SVM
- matplotlib, seaborn
- Streamlit
- joblib
- pytest
- Jupyter Notebook

## Cấu trúc dự án

```text
.
├── app.py                              # Web app nhập dữ liệu và dự đoán
├── Job_level_Classification.ipynb      # EDA, huấn luyện và đánh giá
├── requirements.txt
├── artifacts/
│   ├── job_level_classifier.joblib     # Pipeline đã huấn luyện
│   ├── metrics.json                    # Metric và classification report
│   ├── metadata.json                   # Classes và options cho app
│   └── confusion_matrix.png
├── src/
│   ├── config.py
│   ├── data.py                         # Làm sạch và tạo input
│   ├── modeling.py                     # Khai báo pipeline/mô hình
│   ├── train.py                        # Training entry point
│   ├── predict.py                      # Inference API
│   └── cli.py                          # Command-line prediction
└── tests/
    ├── test_data.py
    └── test_prediction.py
```

## Cài đặt

```bash
git clone <your-repository-url>
cd <repository-name>
python -m venv .venv
```

Windows:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Chạy ứng dụng

Mô hình đã huấn luyện nằm trong `artifacts/`, vì vậy có thể chạy app ngay:

```bash
python -m streamlit run app.py
```

Sau đó mở địa chỉ Streamlit hiển thị trên terminal, thường là `http://localhost:8501`.

## Huấn luyện lại

File dữ liệu gốc không được commit mặc định vì cần xác nhận quyền phân phối. Đặt file `final_project.ods` tại thư mục gốc rồi chạy:

```bash
python -m src.train
```

Lệnh này sẽ:

- làm sạch dữ liệu;
- chia stratified train/test;
- huấn luyện Logistic Regression và Linear SVM;
- so sánh metric;
- chọn mô hình có macro-F1 cao nhất;
- ghi model, metrics, metadata và confusion matrix vào `artifacts/`.

## Dự đoán bằng CLI

```bash
python -m src.cli \
  --title "Senior Software Engineering Manager" \
  --location "Austin, TX" \
  --description "Lead engineers, mentor staff and own project delivery." \
  --function "information_technology_telecommunications" \
  --industry "Information Technology"
```

Ví dụ output:

```text
Manager / Team Leader
1. Manager / Team Leader: 55.68%
2. Specialist: 10.32%
3. Department Head: 9.86%
```

Đây là output thực tế với model hiện tại; kết quả thay đổi theo nội dung input và phiên bản model.

## Chạy kiểm thử

```bash
python -m pytest -q
```

## Hạn chế và hướng phát triển

- Dữ liệu mất cân bằng rất mạnh: lớp lớn nhất có hơn 4.000 mẫu, lớp nhỏ nhất chỉ có 4 mẫu.
- Kết quả của các lớp cực hiếm chưa đủ tin cậy để dùng trong quyết định nhân sự.
- Dữ liệu tin tuyển dụng có thể bị lỗi encoding và thay đổi theo thời gian.
- Relative score của Linear SVM chưa phải calibrated probability.

Hướng cải thiện tiếp theo:

- thu thập thêm mẫu cho các lớp hiếm;
- dùng stratified cross-validation khi số mẫu mỗi lớp đủ lớn;
- tối ưu hyperparameter bằng randomized search;
- thử character n-grams hoặc sentence embeddings;
- hiệu chỉnh xác suất khi có đủ dữ liệu;
- theo dõi data drift và metric theo từng lớp sau triển khai.

## Câu chuyện để trình bày khi phỏng vấn

> Tôi xây dựng một hệ thống NLP phân loại cấp bậc nghề nghiệp từ tin tuyển dụng. Điểm khó nhất là dữ liệu mất cân bằng nghiêm trọng, nên tôi không chỉ dùng accuracy mà chọn macro-F1 làm metric chính. Tôi xây dựng baseline, thử Logistic Regression và Linear SVM, kết hợp TF-IDF với feature phân loại, dùng class weighting và đóng gói preprocessing cùng model trong Pipeline. Sau đó tôi tạo Streamlit app, CLI, test và artifact để biến notebook thành một sản phẩm ML có thể chạy lại.

## License

Source code được phát hành theo MIT License. Dataset không thuộc phạm vi license này; cần kiểm tra quyền sử dụng và phân phối trước khi tải dữ liệu lên GitHub.
