
"""
Software Engineer: Nguyễn Khắc Hưng
Ngày: 30-3-2026
Mô tả: Lớp GARCHGridSearch này là công cụ tự động hóa quá trình lựa chọn mô hình và tinh chỉnh tham số cho họ
mô hình GARCH trong dự án dự báo độ biến động (volatility) của Bitcoin. Khác với GridSearch thông thường trong
học máy vốn chỉ đơn giản là duyệt qua các tổ hợp tham số, lớp này được thiết kế chuyên biệt cho bài toán chuỗi
thời gian tài chính với chiến lược Rolling Forecast — tức là mỗi bước dự báo đều tái huấn luyện lại mô hình
trên toàn bộ dữ liệu đã quan sát được đến thời điểm đó, mô phỏng sát nhất điều kiện dự báo thực tế ngoài
thị trường.

Lớp này đảm nhận ba nhiệm vụ cốt lõi trong pipeline xây dựng Hybrid Model: so sánh hiệu năng giữa ba kiến
trúc GARCH phổ biến nhất trong tài chính định lượng bao gồm GARCH (mô hình hóa phương sai có điều kiện cơ bản),
EGARCH (xử lý tốt hơn với phân phối bất đối xứng của log-return) và GJR-GARCH (mô hình hóa leverage effect —
hiện tượng giá giảm gây ra biến động mạnh hơn giá tăng); tìm kiếm tổ hợp tham số (p, q) và (p, o, q) tối ưu
cho từng kiến trúc thông qua Grid Search toàn diện; và cuối cùng đánh giá định lượng kết quả thông qua ba
chỉ số MSE, RMSE và MAE so sánh với baseline volatility trung bình của tập test.

Kết quả đầu ra của lớp này — bao gồm loại mô hình tối ưu và bộ tham số tốt nhất — sẽ được sử dụng trực tiếp
để huấn luyện mô hình GARCH chính thức, từ đó tạo ra các đặc trưng volatility có cấu trúc thống kê rõ ràng
phục vụ cho bước kết hợp với mô hình Transformer trong kiến trúc Hybrid Model. Việc tự động hóa toàn bộ quá
trình lựa chọn này đảm bảo tính tái lập (reproducibility) và loại bỏ sai lệch do lựa chọn tham số thủ công,
góp phần nâng cao độ tin cậy của toàn bộ hệ thống dự báo.
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
from sklearn.metrics import mean_absolute_error, mean_squared_error


class GARCHGridSearch():
    """
    Grid Search tự động để lựa chọn loại model và tham số tối ưu cho họ mô hình GARCH.

    Lần lượt thực hiện 3 bước:
        1. chon_mo_hinh() — So sánh 3 loại model: GARCH, EGARCH, GJR-GARCH.
        2. chon_tham_so() — Tìm tổ hợp (p, q) tối ưu cho model đã chọn.
        3. in_ra_danh_gia() — In kết quả đánh giá tổng hợp theo MSE, RMSE, MAE.

    Chiến lược dự báo: Rolling Forecast — mỗi bước dự báo t+1 sẽ tái huấn luyện
    model trên toàn bộ dữ liệu train + phần test đã quan sát được đến thời điểm t,
    mô phỏng sát nhất điều kiện dự báo thực tế ngoài thị trường.

    Args:
        return_train     (list): Chuỗi log-return dùng để huấn luyện.
        return_test      (list): Chuỗi log-return dùng để dự báo rolling.
        mean_return_test (float): Giá trị trung bình volatility thực của tập test, dùng làm baseline tham chiếu.
        volatility_test  (list): Giá trị volatility thực của tập test, dùng để tính metric đánh giá.
    """

    def __init__(self, return_train, return_test, mean_return_test, volatility_test):
        self.models = ['Garch', 'EGARCH', 'GJR-GARCH']  # Danh sách model cần so sánh

        self.return_train     = return_train
        self.return_test      = return_test
        self.mean_return_test = mean_return_test
        self.volatility_test  = volatility_test

        # Cấu trúc lưu kết quả: {tên_model: [[dự báo], [MSE, RMSE, MAE], []]}
        self.predict = {
            'Garch'    : [[], [], []],
            'EGARCH'   : [[], [], []],
            'GJR-GARCH': [[], [], []]
        }

    def chon_mo_hinh(self):
        """
        So sánh hiệu năng giữa 3 loại model GARCH với tham số cố định (p=1, q=1).

        Với mỗi model, thực hiện rolling forecast trên 100 mẫu đầu của tập test,
        sau đó tính MSE, RMSE, MAE so với volatility thực để chọn ra loại model
        phù hợp nhất trước khi tinh chỉnh tham số.

        Lưu ý:
            - GJR-GARCH được khởi tạo với tham số bổ sung o=1 để mô hình hóa
              hiệu ứng đòn bẩy (leverage effect) — giá giảm gây biến động mạnh hơn giá tăng.
            - Kết quả được lưu vào self.predict, gọi in_ra_danh_gia() để xem.
        """
        for i in range(3):
            for j in range(100):
                # Mở rộng tập train bằng các mẫu test đã quan sát (rolling)
                train_data = self.return_train + self.return_test[:j]

                if self.models[i] != 'GJR-GARCH':
                    # GARCH và EGARCH: khởi tạo trực tiếp với vol= tên model
                    model = arch_model(
                        train_data,
                        vol=self.models[i],
                        p=1,
                        q=1
                    )
                else:
                    # GJR-GARCH: dùng vol='GARCH' với tham số o=1 (leverage term)
                    model = arch_model(
                        train_data,
                        vol='GARCH',
                        p=1,
                        o=1,
                        q=1
                    )

                model_fit = model.fit(disp='off')
                prediction = model_fit.forecast(horizon=1)
                predict_volatility = np.sqrt(prediction.variance.iloc[-1, 0])  # Chuyển variance → volatility
                self.predict[self.models[i]][0].append(predict_volatility)

            # Tính metric đánh giá sau khi dự báo xong 100 bước
            mse  = mean_squared_error(self.volatility_test, self.predict[self.models[i]][0])
            rmse = np.sqrt(mse)
            mae  = mean_absolute_error(self.volatility_test, self.predict[self.models[i]][0])

            self.predict[self.models[i]][1].extend([mse, rmse, mae])

    def chon_tham_so(self, name_model):
        """
        Grid Search tìm tổ hợp tham số (p, q) tối ưu cho loại model đã chọn.

        Duyệt toàn bộ tổ hợp p ∈ {1,2,3} và q ∈ {1,2,3}, với mỗi tổ hợp
        thực hiện rolling forecast trên toàn bộ tập test và in kết quả MSE,
        RMSE, MAE để so sánh và lựa chọn tham số tốt nhất.

        Lưu ý:
            - GJR-GARCH có thêm vòng lặp o ∈ {1,2,3} để tìm bậc leverage tối ưu,
              dẫn đến tổng cộng 27 tổ hợp thay vì 9.
            - Quá trình này tốn nhiều thời gian do mỗi tổ hợp phải rolling forecast
              trên toàn bộ tập test.

        Args:
            name_model (str): Tên loại model cần tìm tham số.
                              Nhận một trong ba giá trị: 'Garch', 'EGARCH', 'GJR-GARCH'.
        """
        if name_model != 'GJR-GARCH':
            # Grid Search 3x3 = 9 tổ hợp (p, q) cho GARCH và EGARCH
            for i in range(3):
                for j in range(3):
                    predict_value = []

                    for k in range(len(self.return_test)):
                        # Mở rộng tập train theo từng bước rolling
                        train_data = self.return_train + self.return_test[:k]
                        model = arch_model(
                            train_data,
                            vol=name_model,
                            p=i + 1,
                            q=j + 1
                        )
                        model_fit = model.fit(disp='off')
                        forecast  = model_fit.forecast(horizon=1)
                        predict_volatility = np.sqrt(forecast.variance.iloc[-1, 0])  # Chuyển variance → volatility
                        predict_value.append(predict_volatility)

                    # Tính và in metric của tổ hợp (p, q) hiện tại
                    mse  = mean_squared_error(self.volatility_test, predict_value)
                    rmse = np.sqrt(mse)
                    mae  = mean_absolute_error(self.volatility_test, predict_value)
                    print(f"Model: {name_model} (p={i+1}, q={j+1}) | MSE={mse} RMSE={rmse} MAE={mae}")

        else:
            # Grid Search 3x3x3 = 27 tổ hợp (p, o, q) cho GJR-GARCH (có leverage term o)
            for i in range(3):
                for o in range(3):
                    for j in range(3):
                        predict_value = []

                        for k in range(len(self.return_test)):
                            train_data = self.return_train + self.return_test[:k]
                            model = arch_model(
                                train_data,
                                vol='GARCH',
                                p=i + 1,
                                o=o + 1,    # Bậc của leverage term
                                q=j + 1
                            )
                            model_fit = model.fit(disp='off')
                            forecast  = model_fit.forecast(horizon=1)
                            predict_volatility = np.sqrt(forecast.variance.iloc[-1, 0])  # Chuyển variance → volatility
                            predict_value.append(predict_volatility)

                        mse  = mean_squared_error(self.volatility_test, predict_value)
                        rmse = np.sqrt(mse)
                        mae  = mean_absolute_error(self.volatility_test, predict_value)
                        print(f"Model: {name_model} (p={i+1}, o={o+1}, q={j+1}) | MSE={mse} RMSE={rmse} MAE={mae}")

    def in_ra_danh_gia(self):
        """
        In tổng hợp kết quả đánh giá của cả 3 model sau khi chạy chon_mo_hinh().

        Với mỗi model, in ra: MSE, RMSE, MAE và giá trị trung bình volatility
        thực của tập test (baseline tham chiếu) để dễ dàng so sánh mức độ sai số
        tương đối so với biên độ biến động thực tế.

        Lưu ý:
            Cần gọi chon_mo_hinh() trước, nếu không self.predict sẽ rỗng.
        """
        for i in range(3):
            mse  = self.predict[self.models[i]][1][0]
            rmse = self.predict[self.models[i]][1][1]
            mae  = self.predict[self.models[i]][1][2]
            print(
                f"Name model: {self.models[i]} | "
                f"MSE: {mse} | "
                f"RMSE: {rmse} | "
                f"MAE: {mae} | "
                f"Mean volatility: {self.mean_return_test}"
            )