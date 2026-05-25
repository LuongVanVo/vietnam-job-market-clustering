# Giai đoạn 3 — Làm sạch dữ liệu

## 1. Mục tiêu của giai đoạn

Giai đoạn 3 có mục tiêu chuyển đổi dữ liệu thô từ Giai đoạn 2 thành dữ liệu sạch, sẵn sàng cho các bước phân tích khám phá dữ liệu, trích xuất đặc trưng và phân cụm.

Input của giai đoạn này:

```text
raw_data_train.csv
raw_data_test.csv
```

Output của giai đoạn này:

```text
clean_data_train.csv
clean_data_test.csv
```

Sau giai đoạn này, dữ liệu được chuẩn hóa về mặt định dạng, xử lý giá trị thiếu, kiểm tra trùng lặp, chuẩn hóa văn bản, xử lý các biến phân loại, chuẩn hóa thông tin địa điểm/ngành nghề và chuyển đổi cột `salary` từ dạng chuỗi sang các biến có cấu trúc.

---

## 2. Kết quả tổng quan sau làm sạch

Sau khi làm sạch, số lượng bản ghi được giữ nguyên so với dữ liệu thô:

| Tập dữ liệu | Số dòng trước cleaning | Số dòng sau cleaning |
|---|---:|---:|
| Train | 546,190 | 546,190 |
| Test | 60,688 | 60,688 |

Số cột của tập train tăng từ 14 lên 31 cột do bổ sung các biến mới phục vụ phân tích và mô hình hóa:

```text
location_city
location_city_is_unknown
primary_industry

salary_available
salary_min_million_vnd
salary_max_million_vnd
salary_avg_million_vnd
salary_currency
salary_parse_issue

job_title_char_len
job_title_word_count
job_description_char_len
job_description_word_count
requirements_char_len
requirements_word_count
benefits_char_len
benefits_word_count
```

Kết quả kiểm tra ID cho thấy:

```text
overlap_ids_after_clean = 0
```

Điều này chứng minh tập train và tập test không bị trùng bản ghi sau quá trình làm sạch.

---

## 3. Vai trò của làm sạch dữ liệu trong bài toán phân cụm

Trong bài toán phân cụm, chất lượng dữ liệu ảnh hưởng trực tiếp đến chất lượng cụm. Nếu dữ liệu đầu vào bị nhiễu, thiếu, không thống nhất hoặc có nhiều định dạng khác nhau, mô hình có thể nhóm sai các tin tuyển dụng.

Ví dụ:

- Nếu `job_title`, `job_description` hoặc `requirements` bị thiếu mà không xử lý, bước biểu diễn văn bản có thể lỗi hoặc mất thông tin.
- Nếu `salary` có nhiều dạng như “15 - 20 triệu”, “thỏa thuận”, “1000 USD”, “cạnh tranh”, mô hình không thể sử dụng trực tiếp.
- Nếu `location` là địa chỉ chi tiết, mỗi địa chỉ gần như trở thành một giá trị riêng, làm EDA và clustering khó diễn giải.
- Nếu `job_industry` chứa nhiều ngành trong một chuỗi, việc vẽ biểu đồ và phân tích ngành nghề sẽ khó đọc.
- Nếu train và test bị trùng `id`, phần demo trên test sẽ không còn khách quan.

Vì vậy, làm sạch dữ liệu là bước bắt buộc trước khi phân tích và phân cụm.

---

## 4. Nguyên tắc làm sạch dữ liệu

### 4.1. Không xóa dữ liệu quá mạnh

Dữ liệu sau cleaning giữ nguyên số dòng so với dữ liệu thô. Thay vì xóa các dòng thiếu một số trường, nhóm xử lý bằng cách thay thế hoặc đánh dấu giá trị thiếu phù hợp.

Chiến lược chính:

```text
Text missing                    → ""
Categorical missing             → "Unknown"
Salary không parse được         → NaN + salary_available = 0
Salary bất thường               → salary_parse_issue = 1
Location không nhận diện được   → location_city = "Khác/Không rõ"
```

### 4.2. Train và test được xử lý cùng logic

Các hàm cleaning được áp dụng giống nhau cho cả train và test:

```text
raw_train → clean_train
raw_test  → clean_test
```

Điều này giúp quy trình xử lý nhất quán và tránh sai lệch giữa tập huấn luyện và tập kiểm thử.

### 4.3. Tách riêng cleaning và feature engineering

Giai đoạn 3 chỉ làm sạch dữ liệu. Các bước như TF-IDF, TruncatedSVD, One-hot Encoding, Scaling và Clustering chưa được thực hiện ở giai đoạn này. Những bước đó sẽ được triển khai trong các giai đoạn sau.

---

## 5. Các nhóm công việc đã thực hiện

Giai đoạn 3 gồm các nhóm công việc chính:

```text
1. Load raw_data_train.csv và raw_data_test.csv
2. Kiểm tra schema train/test
3. Kiểm tra duplicate và overlap giữa train/test
4. Xử lý missing value
5. Chuẩn hóa các cột văn bản
6. Chuẩn hóa các cột categorical
7. Chuẩn hóa location thành location_city
8. Tạo location_city_is_unknown
9. Chuẩn hóa job_industry thành primary_industry
10. Parse salary thành các biến có cấu trúc
11. Tạo salary_parse_issue để kiểm soát salary bất thường
12. Làm sạch year
13. Tạo các feature độ dài văn bản
14. Kiểm tra dữ liệu sau cleaning
15. Lưu clean_data_train.csv và clean_data_test.csv
16. Lưu metadata, missing summary, salary summary, location/industry coverage và audit files
```

---

## 6. Kiểm tra schema, duplicate và overlap

Trước khi làm sạch, nhóm kiểm tra:

```text
- Train và test có cùng danh sách cột không
- Có dòng trùng lặp toàn bộ không
- Có id bị trùng trong train không
- Có id bị trùng trong test không
- Có id nào xuất hiện ở cả train và test không
```

Kết quả:

```text
same_columns = True
raw_train_duplicate_rows = 0
raw_test_duplicate_rows = 0
raw_train_duplicated_id = 0
raw_test_duplicated_id = 0
overlap_ids_between_raw_train_test = 0
overlap_ids_after_clean = 0
```

Điều này đảm bảo tập test độc lập với tập train, phù hợp yêu cầu của đề bài.

---

## 7. Xử lý missing value

Dữ liệu có tỷ lệ thiếu thấp ở phần lớn cột, nhưng vẫn cần xử lý để đảm bảo các bước tiếp theo không lỗi.

Chiến lược xử lý:

| Nhóm cột | Cột | Cách xử lý |
|---|---|---|
| Text | `job_title`, `job_description`, `requirements`, `benefits` | Thay NaN bằng chuỗi rỗng |
| Categorical | `company_name`, `location`, `job_type`, `job_industry`, `experience_level`, `education_level`, `job_position` | Thay NaN hoặc rỗng bằng `Unknown` |
| Location chuẩn hóa | `location_city` | Không nhận diện được thì gán `Khác/Không rõ` |
| Salary | `salary` | Fill missing bằng `Unknown`, tạo biến cấu trúc |
| Year | `year` | Ép kiểu numeric, giá trị lỗi chuyển thành NaN |

Sau cleaning, các cột text, categorical, `location_city`, `location_city_is_unknown` và `primary_industry` không còn missing. Các cột salary numeric còn missing khoảng 8.33%, điều này là hợp lý vì một số tin không công khai lương hoặc không thể parse thành số.

---

## 8. Chuẩn hóa văn bản

Các cột văn bản chính:

```text
job_title
job_description
requirements
benefits
```

Các bước làm sạch text cơ bản:

```text
- Chuyển NaN thành chuỗi rỗng
- Chuẩn hóa Unicode tiếng Việt
- Chuyển về chữ thường
- Thay xuống dòng/tab bằng khoảng trắng
- Xóa ký tự điều khiển lạ
- Chuẩn hóa khoảng trắng
```

Không xóa toàn bộ tiếng Anh vì trong dữ liệu tuyển dụng có nhiều từ khóa tiếng Anh quan trọng như:

```text
python
java
sql
backend
frontend
react
docker
cloud
sales
marketing
data
developer
```

Các từ khóa này quan trọng cho phân cụm, đặc biệt khi phân biệt nhóm IT, Sales, Marketing, Data hoặc các ngành có thuật ngữ chuyên môn.

---

## 9. Chuẩn hóa categorical features

Các cột categorical chính:

```text
company_name
location
job_type
job_industry
experience_level
education_level
job_position
```

Các bước xử lý:

```text
- Chuẩn hóa Unicode
- Xóa khoảng trắng thừa
- Thay giá trị rỗng bằng Unknown
- Giữ nguyên ý nghĩa gốc của dữ liệu
```

Ở giai đoạn này chưa thực hiện One-hot Encoding hoặc Frequency Encoding. Các kỹ thuật encoding sẽ được thực hiện ở giai đoạn feature engineering.

---

## 10. Chuẩn hóa location thành location_city

Cột `location` gốc là địa chỉ dạng tự do, có thể chứa số nhà, đường, phường/xã, quận/huyện, tỉnh/thành phố hoặc tên tòa nhà. Nếu dùng trực tiếp cột này để EDA, số lượng giá trị duy nhất rất lớn và khó diễn giải.

Vì vậy, nhóm tạo thêm cột:

```text
location_city
```

Cột này được tạo bằng cách nhận diện tên tỉnh/thành phố và một số alias phổ biến trong chuỗi `location`. Các trường hợp không nhận diện được được gán:

```text
Khác/Không rõ
```

Đồng thời, nhóm tạo thêm biến:

```text
location_city_is_unknown
```

Ý nghĩa:

| Cột | Ý nghĩa |
|---|---|
| `location_city` | Tỉnh/thành phố được nhận diện từ chuỗi location |
| `location_city_is_unknown` | Đánh dấu location không nhận diện được tỉnh/thành |

Kết quả coverage trên tập train:

```text
TP Hồ Chí Minh : 285,382 dòng, khoảng 52.25%
Hà Nội         : 141,002 dòng, khoảng 25.82%
Bình Dương     : 25,368 dòng, khoảng 4.64%
Khác/Không rõ  : 13,455 dòng, khoảng 2.46%
Đồng Nai       : 10,150 dòng, khoảng 1.86%
Đà Nẵng        : 8,277 dòng, khoảng 1.52%
Long An        : 7,148 dòng, khoảng 1.31%
```

Tỷ lệ `Khác/Không rõ` khoảng 2.46% là mức chấp nhận tốt đối với dữ liệu địa chỉ dạng tự do. Nhóm không ép tất cả địa chỉ về tỉnh/thành để tránh match sai dữ liệu.

---

## 11. Chuẩn hóa job_industry thành primary_industry

Cột `job_industry` có thể chứa nhiều ngành trong một chuỗi, ví dụ:

```text
Quản lý điều hành , Bán hàng / Kinh doanh , Tiếp thị
```

Để hỗ trợ EDA và feature engineering, nhóm tạo thêm cột:

```text
primary_industry
```

Cách tạo:

```text
primary_industry = ngành đầu tiên trong chuỗi job_industry
```

Ví dụ:

```text
job_industry = "Quản lý điều hành , Bán hàng / Kinh doanh , Tiếp thị"
primary_industry = "Quản lý điều hành"
```

Top ngành chính trên tập train:

```text
Bán hàng - Kinh doanh : 52,977 dòng
Chăm sóc khách hàng   : 50,531 dòng
Kế toán / Kiểm toán   : 45,469 dòng
Xây dựng              : 27,020 dòng
Chưa xác định         : 23,057 dòng
```

Lưu ý: `primary_industry` được dùng để giảm độ phức tạp khi trực quan hóa và diễn giải. Đây không phải hệ thống chuẩn hóa taxonomy ngành nghề tuyệt đối.

---

## 12. Xử lý salary

Cột `salary` là một cột quan trọng nhưng có nhiều định dạng khác nhau.

Ví dụ:

```text
15 - 20 triệu
12.000.000 - 18.000.000 VND
Thỏa thuận
Cạnh tranh
Đang cập nhật
1000 - 1500 USD
0 VND
50.000.000 - 500.000.000 VND
```

Vì vậy, nhóm không sử dụng trực tiếp chuỗi salary ban đầu mà tạo các biến có cấu trúc:

```text
salary_available
salary_min_million_vnd
salary_max_million_vnd
salary_avg_million_vnd
salary_currency
salary_parse_issue
```

Ý nghĩa:

| Cột mới | Ý nghĩa |
|---|---|
| `salary_available` | Tin tuyển dụng có công khai lương cụ thể hay không |
| `salary_min_million_vnd` | Mức lương tối thiểu nếu parse được và là VND |
| `salary_max_million_vnd` | Mức lương tối đa nếu parse được và là VND |
| `salary_avg_million_vnd` | Lương trung bình ước tính |
| `salary_currency` | Đơn vị tiền tệ nhận diện được, ví dụ VND, USD, Unknown |
| `salary_parse_issue` | Đánh dấu salary có khả năng bất thường hoặc gây nhiễu |

Kết quả sau xử lý:

```text
salary_available = 1: 500,840 dòng, khoảng 91.70%
salary_available = 0: 45,350 dòng, khoảng 8.30%
```

Phân bố đơn vị tiền tệ:

```text
VND     : 500,669 dòng
Unknown : 45,350 dòng
USD     : 171 dòng
```

Các dòng salary bất thường được đánh dấu bằng `salary_parse_issue = 1`. Tỷ lệ salary issue trong train khoảng 1.41%, mức chấp nhận được để kiểm soát nhiễu mà không cần xóa dữ liệu.

---

## 13. Kiểm soát salary outlier

Nhóm tạo biến `salary_parse_issue` để đánh dấu các salary bất thường như:

```text
- Lương bằng 0
- Lương trung bình vượt 200 triệu VND/tháng
- salary_min lớn hơn salary_max
- Các chuỗi salary có số nhưng đơn vị/định dạng dễ gây nhiễu
```

Nguyên tắc sử dụng trong các giai đoạn sau:

```text
Khi phân tích lương chính:
salary_available == 1
salary_parse_issue == 0
salary_currency == "VND"
salary_avg_million_vnd >= 1
```

File `stage_03_salary_numeric_describe_no_issue.csv` cho thấy phân phối lương hợp lệ đã ổn định hơn:

```text
salary_avg_million_vnd:
count = 492,981
mean  = 13.56
std   = 7.94
25%   = 9
50%   = 12
75%   = 15.5
max   = 200
```

Điều này cho thấy các outlier salary nghiêm trọng đã được kiểm soát trước khi sang EDA và clustering. Một số giá trị salary rất thấp vẫn có thể tồn tại do dữ liệu gốc có cách ghi theo giờ/ngày/tuần hoặc định dạng không đầy đủ; các giá trị này sẽ được lọc thêm khi phân tích salary ở Giai đoạn 4.

---

## 14. Tạo feature độ dài văn bản

Nhóm tạo một số feature cơ bản về độ dài văn bản:

```text
job_title_char_len
job_title_word_count
job_description_char_len
job_description_word_count
requirements_char_len
requirements_word_count
benefits_char_len
benefits_word_count
```

Các feature này phục vụ:

```text
- EDA
- Phân tích mức độ chi tiết của tin tuyển dụng
- Bổ sung thông tin cho clustering ở giai đoạn feature engineering
```

Độ dài text không quyết định cụm một mình, nhưng là đặc trưng phụ hữu ích.

---

## 15. Kiểm tra sau cleaning

Sau khi làm sạch, nhóm kiểm tra:

```text
- Số dòng train/test trước và sau cleaning
- Số cột trước và sau cleaning
- Missing value sau cleaning
- Overlap ID giữa train/test
- Tỷ lệ salary_available
- Tỷ lệ salary_parse_issue
- Phân phối salary sau khi loại issue
- Coverage của location_city
- Coverage của primary_industry
- Sample text sau cleaning
- Sample salary sau parsing
```

Kết quả:

```text
Số dòng clean_train = số dòng raw_train
Số dòng clean_test = số dòng raw_test
Overlap id giữa train/test = 0
Text/categorical không còn missing
location_city không missing
primary_industry không missing
Salary issue được flag thay vì xóa dòng
```

---

## 16. File đầu ra của giai đoạn 3

File dữ liệu sạch:

```text
data/clean/clean_data_train.csv
data/clean/clean_data_test.csv
```

Copy thêm ở folder gốc để phục vụ nộp bài:

```text
clean_data_train.csv
clean_data_test.csv
```

Các file thống kê và audit:

```text
outputs/tables/stage_03_metadata.csv
outputs/tables/stage_03_duplicate_summary.csv
outputs/tables/stage_03_missing_summary_clean_train.csv
outputs/tables/stage_03_location_city_coverage.csv
outputs/tables/stage_03_unknown_location_sample_100.csv
outputs/tables/stage_03_primary_industry_coverage.csv
outputs/tables/stage_03_salary_available_counts.csv
outputs/tables/stage_03_salary_currency_counts.csv
outputs/tables/stage_03_salary_numeric_describe_all.csv
outputs/tables/stage_03_salary_numeric_describe_no_issue.csv
outputs/tables/stage_03_salary_anomalies.csv
outputs/tables/stage_03_salary_parse_sample_100.csv
outputs/tables/stage_03_text_clean_sample_100.csv
outputs/audit_stage_03/
```

Lưu ý: không sử dụng file cũ `stage_03_salary_numeric_describe.csv` nếu file này vẫn chứa outlier cũ. Trong báo cáo nên dùng `stage_03_salary_numeric_describe_no_issue.csv` cho phân tích lương chính.

---

## 17. Nội dung có thể đưa vào báo cáo

Sau khi chia dữ liệu thành tập huấn luyện và tập kiểm thử, nhóm tiến hành làm sạch dữ liệu nhằm chuẩn bị cho các bước phân tích và phân cụm. Đối với các trường văn bản như `job_title`, `job_description`, `requirements` và `benefits`, nhóm chuẩn hóa Unicode, chuyển chữ về dạng thường, loại bỏ ký tự điều khiển và chuẩn hóa khoảng trắng. Các giá trị thiếu trong trường văn bản được thay bằng chuỗi rỗng để tránh làm mất các bản ghi vẫn còn thông tin hữu ích ở những trường khác.

Đối với các trường phân loại như `company_name`, `location`, `job_type`, `job_industry`, `experience_level`, `education_level` và `job_position`, các giá trị thiếu được thay bằng nhãn `Unknown`. Từ cột `location` gốc, nhóm tạo thêm `location_city` để biểu diễn địa điểm ở cấp tỉnh/thành phố. Các địa chỉ không nhận diện được được gán nhãn `Khác/Không rõ` và đánh dấu bằng `location_city_is_unknown`. Tỷ lệ địa chỉ không nhận diện được chỉ khoảng 2.46% trên tập train. Từ cột `job_industry`, nhóm tạo thêm `primary_industry` bằng cách lấy ngành đầu tiên trong chuỗi ngành nghề nhằm giảm độ phức tạp khi trực quan hóa và diễn giải.

Cột `salary` được xử lý riêng do có nhiều định dạng khác nhau. Nhóm tạo thêm các biến `salary_available`, `salary_min_million_vnd`, `salary_max_million_vnd`, `salary_avg_million_vnd`, `salary_currency` và `salary_parse_issue`. Trong đó, `salary_parse_issue` được dùng để đánh dấu các giá trị lương bất thường như lương bằng 0 hoặc lương trung bình vượt 200 triệu VND/tháng. Các dòng có salary bất thường không bị xóa vì vẫn có thể chứa thông tin hữu ích ở các trường khác.

Sau quá trình làm sạch, số dòng của tập train và test được giữ nguyên. Tập train có 546,190 bản ghi và tập test có 60,688 bản ghi. Không có ID bị trùng giữa hai tập dữ liệu. Dữ liệu sạch gồm 31 cột và được lưu thành `clean_data_train.csv` và `clean_data_test.csv` để sử dụng cho các bước phân tích khám phá dữ liệu, trích xuất đặc trưng và phân cụm ở các giai đoạn tiếp theo.

---

## 18. Nội dung có thể đưa vào slide

### Slide: Làm sạch dữ liệu

```text
Các bước chính:
- Kiểm tra duplicate và overlap giữa train/test
- Xử lý missing value
- Chuẩn hóa text tiếng Việt
- Chuẩn hóa categorical features
- Chuẩn hóa location thành location_city
- Tạo primary_industry từ job_industry
- Parse salary thành các biến có cấu trúc
- Đánh dấu salary bất thường bằng salary_parse_issue
- Tạo text length features
```

### Slide: Chuẩn hóa location và ngành nghề

```text
location gốc:
- Địa chỉ tự do, có số nhà, đường, phường, quận, tỉnh/thành
- Khó trực quan hóa trực tiếp

Biến mới:
- location_city
- location_city_is_unknown

Kết quả:
- TP Hồ Chí Minh: khoảng 52.25%
- Hà Nội: khoảng 25.82%
- Khác/Không rõ: khoảng 2.46%

job_industry:
- Có thể chứa nhiều ngành trong một chuỗi

Biến mới:
- primary_industry
```

### Slide: Xử lý salary

```text
Vấn đề:
salary có nhiều định dạng khác nhau:
- 15 - 20 triệu
- Thỏa thuận
- Cạnh tranh
- 1000 - 1500 USD
- 0 VND
- Giá trị quá lớn bất thường

Biến mới:
- salary_available
- salary_min_million_vnd
- salary_max_million_vnd
- salary_avg_million_vnd
- salary_currency
- salary_parse_issue
```

### Slide: Kết quả cleaning

```text
Train:
546,190 dòng trước cleaning
546,190 dòng sau cleaning

Test:
60,688 dòng trước cleaning
60,688 dòng sau cleaning

Số cột clean data:
31

Overlap ID train/test:
0

Location không nhận diện được:
khoảng 2.46%

Salary issue:
khoảng 1.41% trong train
```

### Slide: Output giai đoạn 3

```text
File dữ liệu sạch:
- clean_data_train.csv
- clean_data_test.csv

File thống kê:
- stage_03_metadata.csv
- stage_03_missing_summary_clean_train.csv
- stage_03_location_city_coverage.csv
- stage_03_primary_industry_coverage.csv
- stage_03_salary_numeric_describe_no_issue.csv
- stage_03_salary_anomalies.csv
```

---

## 19. Điểm cần giải thích khi thuyết trình

### Vì sao không xóa các dòng thiếu benefits hoặc requirements?

Vì các dòng đó vẫn còn nhiều thông tin khác như `job_title`, `job_description`, `salary`, `location`, `job_industry`. Nếu xóa toàn bộ dòng có thiếu một vài trường, dữ liệu sẽ bị mất không cần thiết.

### Vì sao cần tạo `location_city`?

Vì `location` gốc là địa chỉ dạng tự do, có thể chứa số nhà, tên đường, phường, quận và tỉnh/thành. Nếu dùng trực tiếp cột này, số lượng giá trị duy nhất rất lớn và khó trực quan hóa. `location_city` giúp phân tích dữ liệu theo tỉnh/thành phố rõ ràng hơn.

### Vì sao vẫn có `Khác/Không rõ` trong location_city?

Vì không phải tất cả địa chỉ gốc đều chứa đủ thông tin tỉnh/thành. Một số dòng chỉ ghi tên đường, tòa nhà hoặc khu vực. Nhóm không ép các dòng này về một tỉnh/thành để tránh match sai, mà đưa vào nhóm `Khác/Không rõ`.

### Vì sao tạo `primary_industry`?

Vì `job_industry` có thể chứa nhiều ngành trong một chuỗi. `primary_industry` lấy ngành đầu tiên để giảm độ phức tạp khi EDA và diễn giải cụm. Đây không phải chuẩn hóa taxonomy ngành tuyệt đối, mà là đặc trưng hỗ trợ phân tích.

### Vì sao salary cần xử lý riêng?

Vì salary là dữ liệu dạng chuỗi với nhiều định dạng khác nhau. Mô hình không thể hiểu trực tiếp các chuỗi như “15 - 20 triệu”, “thỏa thuận” hoặc “1000 - 1500 USD”. Do đó cần chuyển salary thành các biến có cấu trúc.

### Vì sao tạo `salary_parse_issue` thay vì xóa dòng?

Vì một dòng có salary bất thường vẫn có thể chứa thông tin hữu ích ở các trường khác như job title, mô tả công việc, yêu cầu ứng viên, địa điểm và ngành nghề. Do đó nhóm không xóa dòng mà đánh dấu salary bất thường để xử lý phù hợp ở bước EDA và feature engineering.

### Vì sao chưa làm TF-IDF ở giai đoạn này?

Vì TF-IDF thuộc bước feature engineering. Giai đoạn cleaning chỉ chuẩn bị dữ liệu sạch. Việc tách rõ cleaning và feature engineering giúp quy trình khoa học dữ liệu rõ ràng hơn.

---

## 20. Checklist hoàn thành giai đoạn 3

```text
[x] Load raw_data_train.csv và raw_data_test.csv
[x] Kiểm tra schema train/test
[x] Kiểm tra duplicate toàn dòng
[x] Kiểm tra duplicate theo id
[x] Kiểm tra overlap id giữa train/test
[x] Xử lý missing value cho text columns
[x] Xử lý missing value cho categorical columns
[x] Chuẩn hóa Unicode tiếng Việt
[x] Chuẩn hóa khoảng trắng trong text
[x] Tạo location_city
[x] Tạo location_city_is_unknown
[x] Tạo primary_industry
[x] Parse salary
[x] Tạo salary_available
[x] Tạo salary_min_million_vnd
[x] Tạo salary_max_million_vnd
[x] Tạo salary_avg_million_vnd
[x] Tạo salary_currency
[x] Tạo salary_parse_issue
[x] Làm sạch year
[x] Tạo text length features
[x] Kiểm tra missing sau cleaning
[x] Đảm bảo số dòng không giảm bất thường
[x] Đảm bảo train/test không overlap ID
[x] Lưu clean_data_train.csv
[x] Lưu clean_data_test.csv
[x] Lưu metadata giai đoạn 3
[x] Lưu missing summary sau cleaning
[x] Lưu location/industry coverage
[x] Lưu salary summary và audit files
```

---

## 21. Kết luận giai đoạn 3

Giai đoạn 3 đã hoàn tất. Dữ liệu thô đã được làm sạch và lưu thành `clean_data_train.csv` và `clean_data_test.csv`. Số dòng của train và test được giữ nguyên, không có ID trùng giữa hai tập dữ liệu. Các cột văn bản và categorical đã được xử lý missing, `location` đã được chuẩn hóa thành `location_city`, `job_industry` đã được chuẩn hóa thành `primary_industry`, salary đã được chuyển thành các biến có cấu trúc và các giá trị salary bất thường được đánh dấu bằng `salary_parse_issue`.

Sau giai đoạn này, dữ liệu đã sẵn sàng cho Giai đoạn 4 — Phân tích khám phá dữ liệu và trực quan hóa dữ liệu tuyển dụng.
