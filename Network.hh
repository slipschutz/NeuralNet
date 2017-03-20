

#include "Eigen/Dense"
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXf;
using Eigen::Vector3d;
using Eigen::Vector2d;
using Eigen::Matrix;
using Eigen::Dynamic;

#include <vector>
#include <utility>

using namespace std;

class Network{
public:
  Network(vector <int>);
  ~Network();


  void AddInputNode();

  void SGD(vector <pair<VectorXd,VectorXd> > data,
	   int epochs, int mini_batch_size,double eta,int num=-1);

  void UpdateMiniBatch(vector <pair<VectorXd,VectorXd> > data,
		       vector <VectorXd>&,
		       vector <MatrixXd>&);


  vector <VectorXd> biases2;
  vector <MatrixXd> weights2;


  VectorXd FeedFoward(vector<double>);
  VectorXd FeedFoward(VectorXd);


  void BackProp(vector <double> inputs,
		vector <double> trueAnswer,
		vector <VectorXd> & nb,
		vector <MatrixXd> &nw);

  void BackProp(VectorXd inputs,
		VectorXd trueAnswer,
		vector <VectorXd> & nb,
		vector <MatrixXd> &nw);



  VectorXd cost_derivative(VectorXd output,
			   VectorXd y);
  
private:
  int rNumberLayers;
  vector <int> rSizes;

};


