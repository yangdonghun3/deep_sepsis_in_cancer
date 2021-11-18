# Identification of the risk of sepsis in cancer patients using digital health care records
This project is for the paper "Identification of the risk of sepsis in cancer patients using digital health care records" in the *Journal of the American Medical Informatics Association* (JAMIA).
- Authors of source code: yangdonghun3@kisti.re.kr, jmkim@kisti.re.kr, junnsang@gmail.com, wc.cha@samsung.com, hyojungpaik@gmail.com
- Current version of the project: ver. 0.1

## Abstract
- **Objective:** Sepsis is diagnosed in millions of people every year, resulting in high mortality rate. Although sepsis patients present multimorbid conditions, including cancers, sepsis predictions have mainly focused on patients with severe injuries. Here, we present a machine learning-based approach to identify sepsis risk in cancer patients using electronic health records (EHRs).
- **Materials and Methods:** We utilized anonymized EHRs from the Samsung Medical Center in Korea, including 8,580 cancer patient records in longitudinal manner (2014~2019). To build a prediction model based on physical status that would differ between sepsis and nonsepsis patients, we analyzed 2,462 laboratory (lab) test results and 2,266 medication prescriptions using a graph network and statistical analysis. Based on the results, the model was trained with lab tests and medication relationships.
- **Results:** Sepsis patients showed differential medication trajectories and physical status. For example, in the network-based analysis, narcotic analgesics were prescribed more often in the sepsis group, along with other drugs. Likewise, 35 types of lab test, including albumin, globulin, and prothrombin time, showed significantly different distributions between sepsis and nonsepsis patients (p value<0.0001). Our model outperformed the model trained using only common EHRs, showing improved accuracy, AUROC, and F1-scores up to 11.9%, 11.3%, and 13.6%, respectively. (Accuracy: 0.692, AUROC: 0.753, and F1-score: 0.602 for the random forest-based model).
- **Discussion and Conclusion:** We elucidated that lab tests and medication relationships can be used as efficient features for predicting sepsis in cancer patients. Consequently, identification of sepsis risk in cancer patients using EHRs and machine learning is feasible.

## Requirements
- Python 3.7
- Pytorch 1.5 (GPU version is recomended)
- scipy
- scikit-learn
- shap
- numpy
- pandas
- matplotlib

## License
This project was supported by the Korea Institute of Science and Technology Information (KISTI) (K-21-L02-C10, K-20-L02-C10-S01). 
HP and JK were also supported by the Ministry of Science and ICT (N-21-NM-CA08-S01). 
This research was also supported by the Program of the National Research Foundation (NRF) funded by the Korean government (MSIT) (2021M3H9A203052011). 
The computational analysis was supported by the National Supercomputing Center, including the resources and technology.
- Use of source codes are free for academic researchers. However, the users of source codes from the private sector will need to contact to the developers of the project.
## Caveat
We present the source codes as an example of our research project to help readers understand. We don't provide data preprocessing code. For preprocessing, such as generation drug relationship features, please refer to [our previous work](https://github.com/hypaik/SuicideNetwork).
For testing the proposed spesis prediction model, we provide dummy data instead of the EHR information of cancer patients used in our experiments.

## Experiments
We provide training and testing codes for two machin learning models and three deep learning models for sepsis prediction. 
- Machine learning models: Logistic Regression and Random Forest
- Deep learning models: ANN, ResNet10, and RNN-LSTM
- Each training code (Sepsis_prediction_xxx.ipynb) includes our setup and experimental results.
- The trained models, provided in the "trained_model" directory, can be tested on dummy data using each testing code (Sepsis_prediction_xxx_Test_using_dummy_data.ipynb).    

## Contact information
- Donghun Yang. yangdonghun3@kisti.re.kr
- Jimin Kim. jmkim@kisti.re.kr
- Junsang Yoo. junnsang@gmail.com
- Won Chul Cha. wc.cha@samsung.com
- Hyojung Paik. hyojungpaik@gmail.com
