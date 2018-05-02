// ANN.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#define MIN 0.0001f

using namespace std;

class NeuronLayer {
public:
	//Attributes
	size_t length;
	size_t n_inputs;
	//Neurons
	vector<vector<float>> all_neuron_weights;
	function<const float(const float)> activation_func;
	//Default Constructor
	NeuronLayer() = default;
	//Destructor
	~NeuronLayer() = default;
	//Base Constructor
	NeuronLayer(const size_t& _length, const size_t& _n_inputs, 
		const function<const float(const float)>& _activation_func) :
		length(_length),
		n_inputs(_n_inputs)
	{
		all_neuron_weights.resize(length);
		//Initialize all weights with uniform random values between 0.0 and 1.0
		for (auto& neuron_weights : all_neuron_weights) {
			neuron_weights.resize(n_inputs);
			for (auto& weight : neuron_weights) {
				weight = get_random_weight();
			}
		}

	}
	//Generate uniformly distributed random value from random engine
	float get_random_weight() {
		random_device rd;
		mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
		uniform_real_distribution<> distrib(0.0, 1.0); //Uniform random distribution
		return distrib(gen);
	}
	//Copy Constructor
	NeuronLayer(const NeuronLayer &lhs) :
		//Deep lvalue copy
		length(lhs.length),
		activation_func(lhs.activation_func),
		all_neuron_weights(lhs.all_neuron_weights)
	{
		//empty function body
	}

	//Move Constructor
	NeuronLayer(const NeuronLayer &&rhs) noexcept:
	//Shallow rvalue copy
	length(move(rhs.length)),
		//Fallback to deep copy on container move exceptions
		activation_func(move_if_noexcept(rhs.activation_func)),
		all_neuron_weights(move_if_noexcept(rhs.all_neuron_weights))
	{
		//empty function body
	}

	//(LHS) Overload assignment operator
	NeuronLayer& operator = (const NeuronLayer &lhs) {
		length = lhs.length;
		activation_func = lhs.activation_func;
		all_neuron_weights = lhs.all_neuron_weights;
		return *this;
	}

	//(RHS) Overload assignment operator
	NeuronLayer& operator = (NeuronLayer &&rhs) {
		length = move(rhs.length);
		activation_func = move_if_noexcept(rhs.activation_func);
		all_neuron_weights = move_if_noexcept(rhs.all_neuron_weights);
		return *this;
	}
	vector<float> compute_output(vector<float> inputs) {
		vector<float> outputs;
		//Iterate neurons
		for (auto neuron_weights : all_neuron_weights) {
			//Computer inner product 
			float output =
				inner_product(begin(neuron_weights), end(neuron_weights), begin(inputs), 0.0)
				/ inputs.size();
			outputs.push_back(output);
		}
		
		return outputs;
	}
	//Update weights. Used for back-propagation
	void update_weights(const vector<vector<float>> & new_weights) {
		all_neuron_weights = new_weights;
	}
};
const float sigmoid(const float &input) {
	return 1.0f / (1.0f + exp(-input));
}

class NeuralNet {
public:
	size_t n_hidden_layers;
	NeuronLayer input_layer;
	NeuronLayer hidden_layer;
	NeuronLayer output_layer;
	//Activation function
	function<const float(const float)> sigmoid = 
		[](const float &x) {
			return 1.0f / (1.0f + exp(-x)); 
		};
	//Derivative of the activation function
	function<const float(const float)> sigmoid_prime = 
	[](const float &x) {
		return static_cast<float>(exp(-x)) / pow(1.0f + exp(-x),2.0f); 
	};

	//Constructor
	NeuralNet() {
		//Initialize layers
		hidden_layer = NeuronLayer(3,2,sigmoid);
		output_layer = NeuronLayer(
			1,3, [](const float &x) {return x; }
		);
	}

	//Gradient Descent Back Propagation
	void back_propagate(float output_prime, float output,
		vector<float> hidden_outputs, vector<float> inputs)
	{
		float error = output - output_prime;
		float delta_output = error* sigmoid_prime(
			inner_product(begin(hidden_outputs), end(hidden_outputs),
				begin(output_layer.all_neuron_weights.front()), 0.0)
		);
		//Update Output Layer weights
		vector<vector<float>> new_output_weights = output_layer.all_neuron_weights;
		for (auto& neuron_weights : new_output_weights) {
			for (auto v_iter = neuron_weights.begin(), 
				h_iter = hidden_outputs.begin();
				v_iter != neuron_weights.end();
				++v_iter, ++h_iter)
			{
				//Avoid dividing by zero
				if (*h_iter)
					*v_iter += delta_output / max(*h_iter, MIN);
			}
		}
		output_layer.update_weights(new_output_weights);
		//Computer Hidden Layer Deltas
		vector<float> hidden_deltas;
		for (auto v_iter = output_layer.all_neuron_weights.front().begin(),
					h_iter = hidden_outputs.begin();
			v_iter != output_layer.all_neuron_weights.front().end();
			++v_iter, ++h_iter)
		{
			float denominator = ((*v_iter)*sigmoid_prime(*h_iter));
			float h_delta = delta_output / max(denominator, MIN);
			hidden_deltas.push_back(h_delta);
		}
		vector<vector<float>> new_hidden_weights = hidden_layer.all_neuron_weights;
		//Iterating vector with a traditional loop
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 2; ++j) {
				new_hidden_weights[i][j] = hidden_deltas[i] / max(inputs[j], MIN);
				cout << "h" << hidden_deltas[i] <<"d"<< hidden_deltas[i] / max(inputs[j], MIN) << endl;
			}
		}
		//Update Hidden Layer weights
		hidden_layer.update_weights(new_hidden_weights);
	}


	//Training function
	void train(array<tuple<float, float, float>, 4> training_set) {
		//C++17 auto tie loop
		for (auto[i1, i2, output] : training_set) {
			vector<float> input{ i1,i2 };
			vector<float> hidden_outputs = hidden_layer.compute_output(input);
			float output_prime = (output_layer.compute_output(hidden_outputs)).front();
			back_propagate(output_prime, output, hidden_outputs, input);
		}
	}

	//Prediction function
	float predict(const float i1, const float i2) {
		vector<float> input{ i1,i2 };
		vector<float> hidden_outputs = hidden_layer.compute_output(input);
		//Output normalization
		return (output_layer.compute_output(hidden_outputs)).front();
	}
};


int main()
{
	typedef tuple<float, float, float> triple;
	array<int, 2> x{ 1,2 };
	array<triple, 4> coin_training_set { 
		make_tuple(0,0,1),
		make_tuple(0,1,0),
		make_tuple(1,0,0 ),
		make_tuple(1,1,1)  
	};
	NeuralNet coin_NN;
	coin_NN.train(coin_training_set);
	cout << "Input:0,0" << endl
		<< "Prediction:" << coin_NN.predict(0, 0) << endl;
	cout << "Input:0,1" << endl
		<< "Prediction:" << coin_NN.predict(0, 1) << endl;
	cout << "Input:1,0" << endl
		<< "Prediction:" << coin_NN.predict(1, 0) << endl;
	cout<<"Input:1,1"<<endl
		<<"Prediction:"<<coin_NN.predict(1, 1)<<endl;
	return 0;
}

