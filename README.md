# HRI-ErrorDetector
Code of the work Detecting Interaction Failures through Emotional Feedback and Robot Context

During human-robot interactions, robots may break social norms (Social Norm Violations - SNV) or perform erroneous behaviours due to sensor, actuator, and software errors (Technical Failures - TF). If robots are unaware of these errors, the interaction may become unpleasant or even risk user safety. While interacting, humans show various types of social
signals that translate their inner state. People use these cues to estimate other's inner states, detect interaction issues, and react to them. To detect interaction errors and classify them as Social Norm Violations or Technical Failures, we propose to rely on Eye Gaze, Head Movement, Facial Expressions (Actions Units), and Emotions, from the perspective of the robot, along with the recent actions of the robot. We propose a two step cascaded decision, where the first step is to detect if an error occurs, followed by the error type classification (SNV vs. TF). We perform an extensive study of the various options on input data and classification algorithms, using a game-based scenario with a humanoid robot. 
We use a dataset captured with the Vizzy robot during a block assembly game, composed of 24 individual interactions and two robot moods. The “kind” mood would help the participants win the game. The “grumpy” mood would be rude, causing social norm violations, and would clumsily destroy the assembled blocks, causing technical failures, and making the participant lose the game.
Regarding the impact of input data, we observe that: (i) emotions improve the error detection step but not the error classification step, and (ii) the actions of the robot improves both error detection and error classification. Regarding the learning algorithms, Random Forest achieves the best performance both in error detection and error classification. A median filter on the error classification result increased the performance of Random Forest to 79.63\% mean accuracy.

In the folder DetectErrorCode is located information regarding the dataset used in this work.

There is also the two best trained models for error detection and classification.

You can run CompErrDetect.py to compare our best model to a model that uses features used in previous works.

In the file RecogEmotion.py there's an example of how we obtained emotions from the participants using output from openFace, more detail regarding this can be found in our paper.
