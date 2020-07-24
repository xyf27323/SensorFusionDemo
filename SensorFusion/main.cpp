#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include "Eigen/Eigen"
#include "interface/ground_truth_package.h"
#include "interface/measurement_package.h"
#include "sensorfusion.h"//ͬʱ������kalmanfilter.h

int main(int argc, char* argv[]) {

    // ���ú��ײ��״�/�����״��������ݵ�·��
    // Set radar & lidar data file path
    std::string input_file_name = "../data/obj_pose-laser-radar-synthetic-input.txt";

    // �����ݣ���ʧ�������ʧ����Ϣ������-1������ֹ����
    // Open file. if failed return -1 & end program
    std::ifstream input_file(input_file_name.c_str(), std::ifstream::in);//��ֻ���ķ�ʽ��
    if (!input_file.is_open()) {
        std::cout << "Failed to open file named : " << input_file_name << std::endl;
        return -1;
    }

    // �����ڴ�
    // measurement_pack_list�����ײ��״�/�����״�ʵ�ʲ�õ����ݡ����ݰ�������ֵ��ʱ��������ں��㷨�����롣
    // groundtruth_pack_list��ÿ�β���ʱ���ϰ���λ�õ���ֵ���Ա��ں��㷨�������ֵ�Ĳ�����������ں��㷨����ĺû���
    std::vector<MeasurementPackage> measurement_pack_list;
    std::vector<GroundTruthPackage> groundtruth_pack_list;

    // ͨ��whileѭ�����״����ֵ����ֵȫ�������ڴ棬����measurement_pack_list��groundtruth_pack_list��
    // Store radar & lidar data into memory
    std::string line;
    while (getline(input_file, line)) { //line��inputfile�����ж�ȡ����
        std::string sensor_type;
        MeasurementPackage meas_package;
        GroundTruthPackage gt_package;
        std::istringstream iss(line); //iss���ַ���line�������ȡ����
        long long timestamp;

        // ��ȡ��ǰ�еĵ�һ��Ԫ�أ�L����Lidar���ݣ�R����Radar����
        // Reads first element from the current line. L stands for Lidar. R stands for Radar.
        iss >> sensor_type;
        if (sensor_type.compare("L") == 0) {
            // �����״����� Lidar data
            // ���еڶ���Ԫ��Ϊ����ֵx��������Ԫ��Ϊ����ֵy�����ĸ�Ԫ��Ϊʱ���(���룩
            // 2nd element is x; 3rd element is y; 4th element is timestamp(nano second)
            meas_package.sensor_type_ = MeasurementPackage::LASER;
            meas_package.raw_measurements_ = Eigen::VectorXd(2);
            float x;
            float y;
            iss >> x;
            iss >> y;
            meas_package.raw_measurements_ << x, y;
            iss >> timestamp;
            meas_package.timestamp_ = timestamp;
            measurement_pack_list.push_back(meas_package);
        }
        else if (sensor_type.compare("R") == 0) {
            // ���ײ��״����� Radar data
            // ���еڶ���Ԫ��Ϊ����pho��������Ԫ��Ϊ�Ƕ�phi�����ĸ�Ԫ��Ϊ�����ٶ�pho_dot�������Ԫ��Ϊʱ���(���룩
            // 2nd element is pho; 3rd element is phi; 4th element is pho_dot; 5th element is timestamp(nano second)
            meas_package.sensor_type_ = MeasurementPackage::RADAR;
            meas_package.raw_measurements_ = Eigen::VectorXd(3);
            float rho;
            float phi;
            float rho_dot;
            iss >> rho;
            iss >> phi;
            iss >> rho_dot;
            meas_package.raw_measurements_ << rho, phi, rho_dot;
            iss >> timestamp;
            meas_package.timestamp_ = timestamp;
            measurement_pack_list.push_back(meas_package);
        }

        // ��ǰ�е�����ĸ�Ԫ�طֱ���x�����ϵľ�����ֵ��y�����ϵľ�����ֵ��x�����ϵ��ٶ���ֵ��y�����ϵ��ٶ���ֵ
        // read ground truth data to compare later
        float x_gt;
        float y_gt;
        float vx_gt;
        float vy_gt;
        iss >> x_gt;
        iss >> y_gt;
        iss >> vx_gt;
        iss >> vy_gt;
        gt_package.gt_values_ = Eigen::VectorXd(4);
        gt_package.gt_values_ << x_gt, y_gt, vx_gt, vy_gt;
        groundtruth_pack_list.push_back(gt_package);
    }

    std::cout << "Success to load data." << std::endl;

    // ��ʼ��������㷨
    SensorFusion fuser;
    std::ofstream outputfile("result.txt");//���������
    for (size_t i = 0; i < measurement_pack_list.size(); ++i) {//size_t���Դ洢���������ǿ��ܵ��κ����͵����������С��
        fuser.Process(measurement_pack_list[i]);           //���仰˵��һ��ָ����Ա���ȫ�طŽ�Ϊsize_t����;�����Ϊר��������ʾ����size����������
        Eigen::Vector4d x_out = fuser.kf_.GetX();
        outputfile << x_out(0) << " "<< x_out(1) << " " << x_out(2) << " " << x_out(3) << " "
            <<groundtruth_pack_list[i].gt_values_(0) << " " <<
            groundtruth_pack_list[i].gt_values_(1) << " " <<
            groundtruth_pack_list[i].gt_values_(2) << " " <<
            groundtruth_pack_list[i].gt_values_(3) << std::endl;//ǰ4��Ϊ����ֵ ��4��Ϊ��ֵ
        
     /*   std::cout << "x " << x_out(0) ��ӡ���
            << " y " << x_out(1)
            << " vx " << x_out(2)
            << " vy " << x_out(3)
            << std::endl;
            */
    }
    outputfile.close();
}//main����ʵ�����ݵĶ�ȡ(������ݸ�ʽ�б仯ֻ���main�����õ��������㷨)��������SensorFusion��SensorFusionʵ�ֿ������˲�������ṹ������Э���������������״̬ת�ƾ����
//���������ã�������������б仯ֻ�����SensorFusion����Ӱ��kaiman��Ĺ�ʽ�����������е���kalmanfilter��kalmanfilter�а����˾����Ԥ��͸��¹�ʽ��ʵ�����㷨�ıջ���