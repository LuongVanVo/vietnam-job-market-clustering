# Giai đoạn 4 — EDA và trực quan hóa dữ liệu tuyển dụng bản tối ưu

## Các tối ưu chính

```text
- job_position_display để gom nhãn trùng do khác viết hoa
- education_level_display để dễ giải thích trên biểu đồ
- salary distribution zoom 1-50 triệu
- bỏ Khác/Không rõ khỏi boxplot salary theo tỉnh/thành
- thêm biểu đồ top ngành có lương trung vị cao nhất
- lọc các n-grams bị gãy hoặc quá chung chung
```

## Salary valid filter

```text
salary_available == 1
salary_parse_issue == 0
salary_currency == "VND"
salary_avg_million_vnd >= 1
salary_avg_million_vnd <= 200
salary_min_million_vnd > 0
salary_max_million_vnd > 0
```

## Biểu đồ nên ưu tiên dùng trong báo cáo/slide

```text
stage_04_top_primary_industries.png
stage_04_top_location_cities.png
stage_04_salary_avg_distribution_zoom_1_50.png
stage_04_salary_by_top_primary_industries_boxplot.png
stage_04_top_median_salary_by_primary_industry.png
stage_04_job_description_word_count_distribution.png
stage_04_top_job_title_ngrams.png
stage_04_top_requirements_ngrams.png
```

## Kết luận

Notebook tối ưu này là bản nên dùng để chốt Giai đoạn 4 sau khi chạy lại và kiểm tra output.
