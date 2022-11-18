python main_experiment.py --fairness unconst --attribute 31 --protected_attribute 20 --model_name mobilenet --stratified --number_of_samples 5000
python main_experiment.py --fairness DDP_squared --fairness_parameter 1.0 --attribute 31 --protected_attribute 20 --model_name mobilenet --stratified --number_of_samples 5000
python main_experiment.py --fairness DDP_squared --fairness_parameter 5.0 --attribute 31 --protected_attribute 20 --model_name mobilenet --stratified --number_of_samples 5000
python main_experiment.py --fairness DDP_squared --fairness_parameter 10.0 --attribute 31 --protected_attribute 20 --model_name mobilenet --stratified --number_of_samples 5000
python main_experiment.py --fairness DDP_squared --fairness_parameter 20.0 --attribute 31 --protected_attribute 20 --model_name mobilenet --stratified --number_of_samples 5000
python main_experiment.py --pretrained_on 31 --attribute 20 --fair_backbone unconst  --model_name mobilenet --backbone_stratified --number_of_samples 5000
python main_experiment.py --pretrained_on 31 --attribute 20 --fair_backbone DDP_squared --fair_backbone_parameter 1.0 --model_name mobilenet --backbone_stratified --number_of_samples 5000
python main_experiment.py --pretrained_on 31 --attribute 20 --fair_backbone DDP_squared --fair_backbone_parameter 5.0 --model_name mobilenet --backbone_stratified --number_of_samples 5000
python main_experiment.py --pretrained_on 31 --attribute 20 --fair_backbone DDP_squared --fair_backbone_parameter 10.0 --model_name mobilenet --backbone_stratified --number_of_samples 5000
python main_experiment.py --pretrained_on 31 --attribute 20 --fair_backbone DDP_squared --fair_backbone_parameter 20.0 --model_name mobilenet --backbone_stratified --number_of_samples 5000
python main_experiment.py --attribute 31 --protected_attribute 20 --model_name mobilenet --two_headed --number_of_samples 5000
python learn_explicit_approaches.py --model_name mobilenet --attribute 31 --protected_attribute 20