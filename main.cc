




// class Network(object):

//   def __init__(self, sizes):
//   self.num_layers = len(sizes)
//         self.sizes = sizes
//   self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
//   self.weights = [np.random.randn(y, x) 
// 		  for x, y in zip(sizes[:-1], sizes[1:])]


#include <cmath>
#include <vector>
#include <string>
#include <iostream>
#include <assert.h>
#include <utility>
#include <algorithm>
#include <random>

#include <fstream>
#include <cstdlib>
#include <ctime>

#include "Eigen/Dense"
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXf;
using Eigen::Vector3d;
using Eigen::Vector2d;
using Eigen::Matrix;
using Eigen::Dynamic;

using namespace std;


double BareSigmoid(double z){
  return 1.0/(1.0+exp(-z));
}

double BareSigmoid_Prime(double z){
  return BareSigmoid(z)*(1-BareSigmoid(z));
}


template <typename T>
Matrix<T,Dynamic,1> Vector2Eigen(vector <T> in){

  Matrix<T,Dynamic,1> ret(in.size());
  for (int i=0;i<in.size();i++){
    ret[i]=in[i];
  }
  return ret;

}

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

Network::Network(vector <int> sizes){
  rSizes=sizes;
  //The number of layers will exclude the input layer
  rNumberLayers=sizes.size()-1;
  
  biases2.resize(rNumberLayers);
  for (int i=0;i<rNumberLayers;i++){
    //    biases2[i]=VectorXd::Constant(sizes[i+1],1);
    biases2[i]=VectorXd::Random(sizes[i+1]);
  }
  weights2.resize(rNumberLayers);
  for (int i=0;i<weights2.size();i++){
    ///if the network has first layer with 3 nodes and the 
    //second with 2 the matrix that connects them should look like
    //                      1  1
    //  1  1  1    not      1  1
    //  1  1  1             1  1  
    // so that the when you multiply it with a 3 component vector 
    // you arrive at a 2 component vector which is what the next layer
    // is 
    //    weights2[i]=MatrixXd::Constant(sizes[i+1],sizes[i],1);
    weights2[i]=MatrixXd::Random(sizes[i+1],sizes[i]);
  }


}



Network::~Network(){
  weights2.clear();
  biases2.clear();
  rSizes.clear();
}



void Network::AddInputNode(){
  
// don't need to change the list of biases.
//just need to change the weights to have another
//set of connections of the new node
//   weights2[i]=MatrixXd::Random(sizes[i+1],sizes[i]);

  weights2[0].conservativeResize(weights2[0].rows(), weights2[0].cols()+1);

  VectorXd vec=VectorXd::Random(rSizes[1]);
  weights2[0].col(weights2[0].cols()-1)=vec;
  rSizes[0]=rSizes[0]+1;

}




void Network::SGD(vector < pair < VectorXd,VectorXd > > data,
		  int epochs, int mini_batch_size,double eta,int num){

  int numData = data.size();
  if (num!=-1){
    numData=num;
  }

  for (int i=0;i<epochs;i++){
    std::random_shuffle ( data.begin(), data.end() );
    
    for (int j=0;j<numData/mini_batch_size;j++){


      int offset = j*mini_batch_size;
      vector <pair<VectorXd,VectorXd> > mini_batch(data.begin()+offset,
						   data.begin()+offset+mini_batch_size);

      vector <VectorXd> nabla_b;
      vector <MatrixXd> nabla_w;
      
      UpdateMiniBatch(mini_batch,nabla_b,nabla_w);
      
      ///Now update the weights and biases in the network
      for (int l=0;l<nabla_b.size();l++){
	biases2[l]=biases2[l]-((eta/mini_batch_size)*nabla_b[l]);
	weights2[l]=weights2[l]-((eta/mini_batch_size)*nabla_w[l]);
      }
    }
  }
}

void Network::UpdateMiniBatch(vector <pair<VectorXd,VectorXd> > data,
			      vector <VectorXd> &nabla_b,
			      vector <MatrixXd> &nabla_w){

  for (int i=0;i<data.size();i++){
    vector <VectorXd> nabla_b_delta;
    vector <MatrixXd> nabla_w_delta;
    BackProp(data[i].first,data[i].second,
	     nabla_b_delta,
	     nabla_w_delta);
    
    if (nabla_b.size() == 0){
      //this is the first iteration of the I loop
      //instead of adding the delta Nabla_{b,w}s just
      //set it.  This way the correct dimensions of everything will
      //be there
      nabla_b=nabla_b_delta;
      nabla_w=nabla_w_delta;
    }else{
      //here we add on to the exisiting values in nabla_{b,w}
      for (int j=0;j<nabla_b.size();j++){
	//looping over the layers in the network;
	nabla_b[j]=nabla_b[j]+nabla_b_delta[j];
	nabla_w[j]=nabla_w[j]+nabla_w_delta[j];
      }
    }
  }//end for over the mini batch of data 
  
  //leaving this function the nabla_b and nabla_w should be 
  //ready to change the weights and biases in the network

}

void LoadData(vector <pair <VectorXd,VectorXd> > & data){

  fstream input("./data.txt");
  if (! input.is_open() ){
    cout<<"NO FILE "<<endl;
    return;
  }
  for (int j=0;j<30000;j++){
    VectorXd tempInput(784);
    VectorXd tempOutput(10);
    double v;
    for (int i=0;i<784;i++){
      input>>v;
      tempInput[i]=v;
    }
    string trash;
    input>>trash;
    for (int i=0;i<10;i++){
      input>>v;
      tempOutput[i]=v;
    }
    
    input>>trash;
    data.push_back(make_pair(tempInput,tempOutput));
  }
}







VectorXd Network::FeedFoward(vector <double> vec){
  return FeedFoward(Vector2Eigen(vec));
}
VectorXd Network::FeedFoward(VectorXd temp){
  
  for (int i=0;i<rNumberLayers;i++){
    temp=weights2[i]*temp+ biases2[i];
    temp=temp.unaryExpr(&BareSigmoid);
  }

  return temp;

}



void Network::BackProp(vector <double> inputs,
		       vector <double> trueAnswer,
		       vector <VectorXd> & nabla_b,
		       vector <MatrixXd> & nabla_w){
  BackProp(Vector2Eigen(inputs),Vector2Eigen(trueAnswer),
	   nabla_b,nabla_w);


}


void Network::BackProp(VectorXd inputs,
		       VectorXd trueAnswer,
		       vector <VectorXd> & nabla_b,
		       vector <MatrixXd> & nabla_w){

  //need to store the nabla_b's and nabla_w's
  //these need to be the same sizes as the biases and 
  //weight vectors


  nabla_b.resize(biases2.size());
  for (int i=0;i<nabla_b.size();i++){
    nabla_b[i]=VectorXd::Zero(biases2[i].size());
  }


  nabla_w.resize(weights2.size());
  for (int i=0;i<nabla_w.size();i++){
    nabla_w[i]=MatrixXd::Zero(weights2[i].rows(),
			      weights2[i].cols());
  }
  

  
  VectorXd  activation = inputs;
  vector<VectorXd>  activations;

  activations.push_back(activation);

  vector<VectorXd> zs;
  ///need to do similar feed foward as in the feed foward method
  for (int l=0;l<rNumberLayers;l++){

    VectorXd z = weights2[l]*activation+biases2[l];
    zs.push_back(z);
    activation = z.unaryExpr(&BareSigmoid);
    activations.push_back(activation);
  }

  int sAct=activations.size();
  VectorXd delta=cost_derivative(activations[sAct-1],trueAnswer)
    .cwiseProduct(zs[zs.size()-1].unaryExpr(&BareSigmoid_Prime));



  ///Set the vector for the last layer in nabla_b to delta
  nabla_b[nabla_b.size()-1]=delta;

  nabla_w[nabla_w.size()-1]=delta*activations[sAct-2].transpose();


  for (int l=1;l<rNumberLayers;l++){
    int theLayer=nabla_w.size()-1-l;

    VectorXd z = zs[theLayer];
    VectorXd sp=z.unaryExpr(&BareSigmoid_Prime);

    delta=weights2[theLayer+1].transpose()*delta;
    delta=delta.cwiseProduct(sp);

    nabla_b[theLayer]=delta;
    nabla_w[theLayer]=delta*activations[theLayer].transpose();
  }

}

VectorXd Network::cost_derivative(VectorXd output,
					VectorXd y){
  return output-y;

}

int GetNumber(VectorXd vec){

  double max =-999999;
  int maxIndex=-1;
  for (int i=0;i<vec.size();i++){
    if (vec[i] >max){
      max =vec[i];
      maxIndex=i;
    }
  }
  return maxIndex;
}


int main(int argc, char ** argv){
  srand((unsigned int) time(0));
  
//   MatrixXd mat=MatrixXd::Constant(3,3,1);
//   cout<<mat<<endl;
//   VectorXd vec=VectorXd::Constant(4,10);
//   mat.conservativeResize(mat.rows(), mat.cols()+1);
//   mat.col(mat.cols()-1) = vec;
//   cout<<mat<<endl;

  if (argc != 4){
    cout<<"Give arguments"<<endl;
    return -1;
  }

  int mSize=atoi(argv[1]);
  double eta=atof(argv[2]);
  int epochs = atoi(argv[3]);

  vector < pair <VectorXd,VectorXd> > allData;
  cout<<"Loading data..."<<endl;
  LoadData(allData);
  int trainingSize=floor(0.8*allData.size());

  vector < pair <VectorXd,VectorXd> > trainingData(allData.begin(),allData.begin()+trainingSize);
  vector < pair <VectorXd,VectorXd> > testData(allData.begin()+trainingSize,allData.end());

  cout<<"Done"<<endl;

  


  Network A({783,30,10});
  //data epoch minisize eta
  A.AddInputNode();


  cout<<"Training..."<<endl;
  A.SGD(trainingData,epochs,mSize,eta);
  cout<<"Done "<<endl;


  int totalRight=0;
  for (int i=0;i<testData.size();i++){

    auto ans = A.FeedFoward(testData[i].first);

    if (GetNumber(testData[i].second) == GetNumber(ans) ){
      totalRight++;
    }
  }
  cout<<"total right "<<totalRight<<" "<<totalRight/double(testData.size())<<endl;

  return 0;
}
