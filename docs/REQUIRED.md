Phân cụm bộ dữ liệu việc làm Việt Nam.
- Input: toàn bộ dataset Tinix Vietnam Job Description.
Link: https://huggingface.co/datasets/tinixai/vietnamese-job-descriptions?fbclid=IwY2xjawReSItleHRuA2FlbQIxMABicmlkETFMdnV6cWRZZWxPQnRDWldDc3J0YwZhcHBfaWQQMjIyMDM5MTc4ODIwMDg5MgABHjHX7IzdvIDJuhvemwHSWzyZu8c314CdPSjWRI8u8BkxS5_OVKSg3b2OhKCP_aem_rk-hwsxKBue3L8w_dzoJ7A
- Output: số cụm và hiệu suất phân cụm của thuật toán (dùng các metrics phổ biến của BT phân cụm), các thuộc tính chung (phổ biến) của các phần tử thuộc mỗi cụm và dựa vào đó đưa ra nhãn (tự gán) cho mỗi cụm.
Dự đoán dữ liệu dùng mô hình chuỗi thời gian.
- dataset: nhóm tự thu thập dữ liệu chuỗi thời gian.
- yêu cầu: Biến mục tiêu tự chọn. PHẢI có dùng biến ngoại sinh trong mô hình dự báo theo một cách nào đó. 


Bài làm cần upload gồm 01 folder chứa: file PDF của quyển báo cáo, file PDF của SLIDE, 01 (hoặc nhiều) file mã nguồn Jupyter notebook dùng để thực hiện các tác vụ, file README ghi hướng dẫn trình tự chạy chương trình, 02 file (hoặc thư mục con) chứa dữ liệu thô (đặt tên là "raw_data_train.csv" và "raw_data_test.csv"), 02 file (hoặc thư mục con) chứa dữ liệu sau khi đã làm sạch (ví dụ: loại bỏ các lỗi định dạng, xoá các ký tự không liên quan,…) trước khi thực hiện các bước feature engineering (đặt tên là "clean_data_train.csv" và "clean_data_test.csv"), 02 file training.ipynb và testing.ipynb.


Trình bày SLIDE theo thứ tự nội dung của "Mau tieu luan KHDL_2026": nhóm tự sắp xếp trình bày theo SLIDE theo phân công nhiệm vụ của mỗi SV trong tối đa 10 phút, sau đó chạy demo testing.ipynb trong tối đa 5 phút để giải thích kết quả. SV tập trung trình bày về các đặc tính của dữ liệu và đưa ra các nhận xét kèm theo lý giải, rút ra kết luận về các kỹ thuật đóng góp vào cải thiện hiệu quả của bài toán (dùng bảng và đồ thị để dễ trình bày về dữ liệu và so sánh đối chiếu các kết quả). Mỗi nhóm chuẩn bị sẵn 2-3 máy tính để phòng sự cố.