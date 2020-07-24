#include "sensorfusion.h"//ͬʱ������kalmanfilter.h


SensorFusion::SensorFusion()
{
    is_initialized_ = false;
    last_timestamp_ = 0.0;

    // ��ʼ�������״�Ĳ������� H_lidar_
    // Set Lidar's measurement matrix H_lidar_
    H_lidar_ = Eigen::MatrixXd(2, 4);    //MatrixXd����ʾ�����С��Ԫ������Ϊdouble�ľ�����������Сֻ��������ʱ����ֵ֮�����֪����
    H_lidar_ << 1, 0, 0, 0,
        0, 1, 0, 0;

    // ���ô������Ĳ�����������һ���ɴ����������ṩ���粻�ṩ��Ҳ��ͨ���о���Ĺ���ʦ���Եõ�
    // Set R. R is provided by Sensor supplier, in sensor datasheet
    // set measurement covariance matrix
    R_lidar_ = Eigen::MatrixXd(2, 2);
    R_lidar_ << 0.0225, 0,
        0, 0.0225;

    // Measurement covariance matrix - radar
    R_radar_ = Eigen::MatrixXd(3, 3);
    R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;
}

SensorFusion::~SensorFusion()
{

}

void SensorFusion::Process(MeasurementPackage measurement_pack)
{
    // ��һ֡�������ڳ�ʼ�� Kalman �˲���
    if (!is_initialized_) {
        Eigen::Vector4d x;
        if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
            // �����һ֡�����Ǽ����״����ݣ�û���ٶ���Ϣ����˳�ʼ��ʱֻ�ܴ���λ�ã��ٶ�����Ϊ0
            x << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
        }
        else if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
            // �����һ֡�����Ǻ��ײ��״����ͨ�����Ǻ������x-y����ϵ�µ�λ�ú��ٶ�
            float rho = measurement_pack.raw_measurements_[0];
            float phi = measurement_pack.raw_measurements_[1];
            float rho_dot = measurement_pack.raw_measurements_[2];
            float position_x = rho * cos(phi);
            if (position_x < 0.0001) {
                position_x = 0.0001;
            }
            float position_y = rho * sin(phi);
            if (position_y < 0.0001) {
                position_y = 0.0001;
            }
            float velocity_x = rho_dot * cos(phi);
            float velocity_y = rho_dot * sin(phi);
            x << position_x, position_y, velocity_x, velocity_y;
        }

        // ��������ʱ��0��Ϊ������
        if (fabs(x(0)) < 0.001) {
            x(0) = 0.001;
        }
        if (fabs(x(1)) < 0.001) {
            x(1) = 0.001;
        }
        // ��ʼ��Kalman�˲���
        kf_.Initialization(x);

        // ����Э�������P
        Eigen::MatrixXd P = Eigen::MatrixXd(4, 4);
        P << 1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1000.0, 0.0,
            0.0, 0.0, 0.0, 1000.0;
        kf_.SetP(P);

        // ���ù�������Q
        Eigen::MatrixXd Q = Eigen::MatrixXd(4, 4);
        Q << 1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0;
        kf_.SetQ(Q);

        // �洢��һ֡��ʱ���������һ֡����ʹ��
        last_timestamp_ = measurement_pack.timestamp_;
        is_initialized_ = true;
        return;
    }

    // ��ǰ����֡��ʱ�����ݰ��е�ʱ�����λΪ΢�룬����1e6��ת��Ϊ��
    double delta_t = (measurement_pack.timestamp_ - last_timestamp_) / 1000000.0; // unit : s
    last_timestamp_ = measurement_pack.timestamp_;

    // ����״̬ת�ƾ���F
    Eigen::MatrixXd F = Eigen::MatrixXd(4, 4);
    F << 1.0, 0.0, delta_t, 0.0,
        0.0, 1.0, 0.0, delta_t,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0;
    kf_.SetF(F);

    // Ԥ��
    kf_.Prediction();

    // ����
    if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
        kf_.SetH(H_lidar_);
        kf_.SetR(R_lidar_);
        kf_.KFUpdate(measurement_pack.raw_measurements_);
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
        kf_.SetR(R_radar_);
        // Jocobian����Hj�������Ѱ�����EKFUpdate��
        kf_.EKFUpdate(measurement_pack.raw_measurements_);
    }
}
