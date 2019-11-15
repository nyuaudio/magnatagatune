# magnatagatune
<b>Overview</b><br>
Our group addressed problem of musical genre classification, which is a subset of audio classifcation in general. <br> <br>

We utilized transfer learning to train a convolutional neural network for a multi-label classification, using each audio sample’s mel spectrogram images as input features. The model was based on the VGG model trained for audio files with ‘audioset’, namely “VGGish”. Our final testing accuracy across all genres was 98%. <br>

 <p align="center">
 <b> Final Training and Test Accuracy </b>
 </p>
<p align="center">
<img width="400" height="300" src="https://raw.githubusercontent.com/tommy-fox/magnatagatune/master/accuracy_plot.png">
</p>
<br>

 <p align="center">
 <b> Optimal Model Summary </b>
 </p>
 <p align="center">
 <img width="450" height="600" src="https://raw.githubusercontent.com/tommy-fox/magnatagatune/master/model_summary.png"> 
 </p>
 <br>

<b> Data and Preprocessing </b><br>
We used the magnatagatune dataset, obtained from the MIREX website, which consists of over 20,000 mono audio files each 30 seconds long. We used 75% of the data for training and 25% for testing. The dataset consisted of raw audio files, classified into 188 genre classes specified in a CSV file. <br><br>

We used Librosa to load audio files, then divided each sample into 30 roughly 1-second(0.96 seconds) segments. From this, we extracted Mel spectrograms, designating resolution to be 64 frequency bins and 96 time frames. As a result, the features extracted for each audio clip are a tensor of (30,96,64). <br><br>
Finally we randomly selected one (96,64) slice out of this tensor and designated it as the input to the network. To improve training speed, we preprocessed all audio files and stored the Mel spectrograms along with the class label in Pickle files on our virtual machine. <br><br>

<b> Training </b><br>
We trained the network using a GPU installed on a virtual machine instance provided by the Google Cloud Platform. <br>
The feature extraction method closely follows the ​VGGish network​ Github repository, since the extraction methods are already tuned to the network. <br> <br>
 
In order to use batch processing, we implemented a custom data generator with pescador,
which multiplexed 20 active streamers that stochastically fetched spectrograms from our data subsets.
With 12 epochs and batch size of 32, the model was trained iteratively using the adam optimizer and binary cross entropy loss function. <br><br>

<b> Experiments </b><br>
In our preliminary experiment, we added 2 sets of batch-normalization + drop-out + fully connected layers on top of the VGGish model (one using a rectified linear unit activation function and the ending dense layer using softmax activation function).
Six convolutional layers and 4 max pooling layers are interlaid in the VGGish model, followed by a global average pooling layer. <br><br>

The accuracy of this model started near 98% and did not stray very far from this throughout training.
We hypothesize that the model was fairly accurate at the onset due to VGGish having already been trained on a large number of audio signals.
Overall loss of the model kept decreasing until around the 9th epoch. <br> <br>

The preliminary training used training set size / batch size steps per epoch, which exceeds the actual number of pickle files we generated, as we found out later.
Thus, the training session performed more iterations of optimizations and the loss converged more noticeably. <br><br>

One odd result from this model is that the validation accuracy is constantly higher than accuracy.
After researching, we found out that the addition of dropout layers prompted this to happen, as all features being used during test and yields more robust results.<br>

In following experiments, we revised the steps per epoch to actual number of pickle files divided by batch size. 
The accuracy was still around 0.9818 and loss became stagnant around 9th epoch. <br><br>

After that, we first tried adding the third set of batch-normalization + dropout + fully connected layers, and training the model only for 5 epochs.
Comparing with the first 5 epochs of previous models, the result hasn’t improved much. Loss even converged slightly slower than training with 2 sets of regularization layers, however test accuracy was able to be kept above 0.9819. <br>

We ultimately tried using the ​autopool​ layer to replace the last global averaging pooling layer in VGGish.
 
