# Machine_Learning-CMPT353_e7
<h3>This repo is created for documentation purpose. The repo contains my personal work toward the SFU CMPT353 (Computational Data Science) course. You may use my solution as a reference. The .zip archive contains the original exercise files. For practice purpose, you can download the .zip archive and start working from there.</h3>

<p><a href="https://coursys.sfu.ca/2018su-cmpt-353-d1/pages/AcademicHonesty">Academic Honesty</a>: it's important, as always.</p>

<br/>
<p>There are 4 tasks in this exercise. </p>
<p>First task: revisit exercise 2; but this time, you will need to get the p-value from the regression and plot a histogram. </p>
<p>Second task: revisit exercise 3; you are given more CPU data and you are expected to use and train a linear regression model to fit the CPU temperature data. </p>
<p>Third task: you are given colour data in the form of RGB values. You will first need to split your data into two groups: train and test. Afterwards, you will train a Naive Bayes (NB) classifier which will be later used to help you predict the colors' labels. </p>
<p>TFourth task: to improve the ML model's accuracy, you need to convert the RGB data to LAB data and build a new pipeline model. Numpy reshaping, FunctionTransformer, rgb2lab, and many techniques are involved in this task.</p>
<br/>

<p>Below is the exercise description </p>
<hr>


<h2 id="h-dog-rates-significance">Dog Rates Significance</h2>
<p>One last statistics question: when we looked at <span>&ldquo;</span>Pup Inflation<span>&rdquo;</span> in <a href="Exercise2">Exercise 2</a>, we drew a fit line across the data, but didn't really ask the question at hand: have the ratings given by this Twitter account been changing over time?</p>
<p>Revisit your <code>dog-rates.ipynb</code> from exercise 2 and append some more useful results. <strong>Output the p-value</strong> from the regression for the question <span>&ldquo;</span>is the slope different from zero?<span>&rdquo;</span>. Also <strong>plot a histogram</strong> of the residuals (observed values minus predicted values). [Note the question about this below.]</p>
<h2 id="h-cpu-temperature-regression">CPU Temperature Regression</h2>
<p>Let's revisit another question from the past. When we looked at CPU temperature values in <a href="Exercise3">Exercise 3</a>, we made a prediction for the next temperature: could we have done better?</p>
<p>There was more data in my original data set. I actually collected CPU usage percent (at the moment the sample was taken), the <a href="https://en.wikipedia.org/wiki/Load_(computing)">system load</a> (one minute average), fan speed (RPM), and CPU temperature. Also, data was collected every 10 seconds: the Exercise<span>&nbsp;</span>3 data set was subsampled down to once per minute. Surely we could have made a better <span>&ldquo;</span>CPU temperature in the next time step<span>&rdquo;</span> prediction with more data.</p>
<p>For this question, I have provided the expanded data set as <code>sysinfo.csv</code>.</p>
<p>Create a program <code>regress_cpu.py</code> based on the provided <code>regress_cpu_hint.py</code>. Things you need to do:</p>
<ul><li>Fill in the <code>'next_temp'</code> column in the DataFrame when it's read. This should be the temperature at time <em>t</em>+1, which is what we want to predict. <a href="http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.shift.html">Hint</a>.
</li><li>Create a scikit-learn linear regression model and fit it to the training data. <a href="http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html">Hint</a>, also do <strong>not</strong> fit an intercept (since there's nowhere to put it in the Kalman filter).
</li><li>Update the <code>transition</code> matrix for the Kalman filter to actually use the new-and-improved predictions for temperature.
</li></ul>
<p>Have a look at the output and note the questions below. When you submit, the <strong>output should be</strong> the one line from the <code>output_regression</code> function provided in the hint.</p>
<h2 id="h-colour-words">Colour Words</h2>
<p>Thanks to all of you for contributing data mapping colours (specifically <a href="https://en.wikipedia.org/wiki/RGB_color_model">RGB colours</a> you saw on-screen) to colour words. When creating the experiment, I gave options for the English <a href="https://en.wikipedia.org/wiki/Color_term">basic colour terms</a>.</p>
<p>The result has been a nice data set: almost 4000 data points that we can try to learn with. It is included this week as <code>colour-data.csv</code>.</p>
<p>Let's actually use it for its intended purpose: training a classifier.</p>
<p>Create a program  <code>colour_bayes.py</code>. You should take the name of the CSV file on the command line: the provided <code>colour_bayes_hint.py</code> does this and contains some code to nicely visualize the output.</p>
<p>Start by getting the data: read the CSV with Pandas. Extract the <code>X</code> values (the R, G, B columns) into a NumPy array and normalize them to the 0<span>&ndash;</span>1 range (by dividing by 255: the tools we use will be looking for RGB values 0<span>&ndash;</span>1). Also extract the colour words as <code>y</code> values.</p>
<p>Partition your data into training and testing sets using <a href="http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html">train_test_split</a>.</p>
<p>Now we're ready to actually do something: create a
<a href="http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB">naïve Bayes classifier</a> and train it. Use the default priors for the model: they are set from the frequency in the input, which seems as sensible as anything else.</p>
<p>Have a look at the accuracy score on the test data to see how you did. Print the accuracy score for this model.</p>
<p>The score doesn't tell much of a story: call <code>plot_predictions</code> from the hint to see a plot of colours (left) and predicted colour categories (right).</p>
<h2 id="h-colour-words-and-colour-distances">Colour Words and Colour Distances</h2>
<p>The naïve Bayes approach implicitly assumes that distances in the input space make sense: distances between training <code>X</code> and new colour values are assumed to be comparable. That wasn't a great assumption: <a href="https://en.wikipedia.org/wiki/Color_difference">distances between colours</a> in RGB colour space aren't especially useful.</p>
<p>Possibly our inputs are wrong: the <a href="https://en.wikipedia.org/wiki/Lab_color_space">LAB colour space</a> is much more perceptually uniform. Let's convert the RGB colours we have been working with to LAB colours, and train on that.
The <a href="http://scikit-image.org/docs/dev/api/skimage.color.html">skimage.color</a> module has a function for the conversion we need. (You may have to <a href="InstallingPython">install</a> scikit-image, depending on your original setup).</p>
<p>We can create a <a href="http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html#sklearn.pipeline.make_pipeline">pipeline</a> model where the first step is a transformer that converts from RGB to LAB, and the second is a Gaussian classifier, exactly as before.</p>
<p>There is no built-in transformer that does the colour space conversion, but if you write a function that converts your <code>X</code> to LAB colours, you can create a <a href="http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html#sklearn.preprocessing.FunctionTransformer">FunctionTransformer</a> to do the work.</p>
<p>The skimage.color functions assume you have a 2D image of pixel colors: you will have to a little <a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html">NumPy reshaping</a> in your function to make it all work. Reshape the array of colours to an image \(1\times n\) (which is a <code>.reshape(1,-1,3)</code>), convert to LAB, and then reshape back to an array of colour values (<code>.reshape(-1,3)</code>).</p>
<p>Have a look at the accuracy value for this model as well. When finished, <strong>your <code>colour_bayes.py</code> should print two lines</strong>: the first and second accuracy score. Please do <strong>not</strong> have a <code>plt.show()</code> in your code when you submit: it makes marking a pain.</p>
<h2 id="h-questions">Questions</h2>
<p>Answer these questions in a file <code>answers.txt</code>.</p>
<ol><li>Looking at your <code>dog-rates.ipynb</code>, do you think the residual are close-enough to being normal to look at the OLS p-value? Can you reasonably conclude that the ratings are increasing?
</li><li>Do you think that the new <span>&ldquo;</span>better<span>&rdquo;</span> prediction is letting the Kalman filter do a better job capturing the true signal in the noise?
</li></ol>
