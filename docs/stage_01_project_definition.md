# Giai đoạn 1 — Xác định bài toán phân cụm

## Tên đề tài

**Phân cụm tin tuyển dụng tại Việt Nam dựa trên mô tả công việc, yêu cầu ứng viên và đặc điểm tuyển dụng**

## Loại bài toán

Đây là bài toán học không giám sát, cụ thể là phân cụm dữ liệu tuyển dụng. Mục tiêu là tìm các nhóm tin tuyển dụng có đặc điểm tương đồng mà không cần nhãn cụm được gán sẵn.

## Dataset

Dataset sử dụng: `tinixai/vietnamese-job-descriptions`.

Các nhóm thông tin chính gồm mô tả công việc, yêu cầu ứng viên, quyền lợi, mức lương, địa điểm, hình thức làm việc, ngành nghề, vị trí, công ty và năm đăng tin.

## Phạm vi dữ liệu

Dùng toàn bộ dataset TiniX. Chia dữ liệu theo yêu cầu mới: 90% train và 10% test. Tập test không trùng tập train.

## Input

Input gồm `job_title`, `job_description`, `requirements`, `benefits`, `salary`, `location`, `job_type`, `job_industry`, `experience_level`, `education_level`, `job_position`, `company_name`, `year`.

## Output

Output là `cluster_id`. Sau khi phân cụm, mỗi cụm sẽ được diễn giải bằng top keywords, ngành nghề phổ biến, mức lương, kinh nghiệm, địa điểm và hình thức làm việc.

## Câu hỏi nghiên cứu

1. Có thể chia tin tuyển dụng thành bao nhiêu nhóm chính?
2. Mỗi cụm đại diện cho nhóm công việc nào?
3. Cụm nào có lương trung vị cao hơn?
4. Cụm nào yêu cầu kinh nghiệm cao hơn?
5. Cụm nào tập trung ở Hà Nội, TP.HCM hoặc địa phương khác?
6. Các từ khóa đặc trưng của từng cụm là gì?

## Pipeline

Raw data → Train/test split 90/10 → Data cleaning → Text cleaning → Salary parsing → TF-IDF → TruncatedSVD/LSA → Structured feature encoding → Clustering → Evaluation → Interpretation → Demo 10 mẫu test.

## Nội dung báo cáo

Đề tài tập trung vào bài toán phân cụm tin tuyển dụng tại Việt Nam dựa trên mô tả công việc, yêu cầu ứng viên và các đặc điểm tuyển dụng. Đây là bài toán học không giám sát, trong đó dữ liệu không có nhãn cụm ban đầu. Mục tiêu của đề tài là phát hiện các nhóm tin tuyển dụng có đặc điểm tương đồng, từ đó hỗ trợ phân tích cấu trúc thị trường tuyển dụng và nhận diện các nhóm công việc phổ biến.
