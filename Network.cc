

#include "Network.hh"

#include <cmath>

#include <string>
#include <iostream>
#include <assert.h>

#include <algorithm>
#include <random>


#include <cstdlib>
#include <ctime>





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

