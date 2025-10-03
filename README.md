The project contains three parts:

1. Encoding part://
   We use brian2 to accomplish the simulation of one-layer neuronal network.
   Specifically, we use LIF model to describe a single neuron's dynamics, and use
   
2. Decoding part:
   
   We use a neural network as decoder.
   This part aims to simulate the process in our brain that receive spike trains and generate image in our mind.
   The spike train vector will first go through a 2-layer fully connected neural network and then becomes the same size as the flattened image,
   and then go through a U-Net to reconstruct the targeted image.
   The framework follows Liu.
   
4. Demo
   This part of code is mainly completed by my senior, Ming Zhang.
   By using a computer camera, one can capture a real-life photo, resize it,
   and then obtain the corresponding spike train as well as the reconstructed image.
