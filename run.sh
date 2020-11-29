data_type="" # your data name
input_dim=300
output_dim=300
hidden_dim=64
ns_amount=10
learning_rate=0.0001
drop_rate=0.01
batch_size=32
epochs=100
output_length=1000

args="--data_type ${data_type} --input_dim ${input_dim} --output_dim ${output_dim} --hidden_dim ${hidden_dim} --ns_amount ${ns_amount} --learning_rate ${learning_rate} --drop_rate ${drop_rate} --batch_size ${batch_size} --epochs ${epochs} --output_length ${output_length}"
echo $args

# Training
python word2vec.py --data_type $data_type --input_dim $input_dim
python Model.py $args
python generate_ltr_data.py $args
python generate_ltr_testdata.py $args

# Learning to rank
model_type="lambdaMART"
ranklib_path="/path/to/ranklib.jar"
python learning2rank.py --data_type $data_type --model_type $model_type --ranklib_path $ranklib_path --pred

# Evaluation
python evaluation.py --data_type $data_type --model_type $model_type
