📄 Augmentation Report
1. Objective

The goal of this stage is to increase the diversity of the dataset and reduce the risk of overfitting during model training.
By applying augmentation techniques, we ensure that the model sees a wider variety of hand gesture images, making it more robust and generalizable.

2. Implemented Augmentations

The following augmentations were applied using Keras ImageDataGenerator inside the pipeline (augmentation.py):

Rotation (±20°): Simulates different hand orientations while avoiding unnatural distortions.

Width & Height Shift (10%): Simulates changes in the hand’s position within the frame.

Zoom (±20%): Mimics variations in camera distance from the hand.

Shear (0.2): Adds small distortions to simulate realistic variations.

Horizontal Flip: Creates mirrored versions of gestures to double the dataset variety.

Brightness Range [0.8 – 1.2]: Handles different lighting conditions.

Normalization (rescale=1./255): Ensures pixel values are between 0 and 1 for stable training.

3. Experiments & Visualization

A Jupyter notebook (augmentation_demo.ipynb) was created to demonstrate and visualize the augmentations:

One random training image was selected.

Augmentation was applied dynamically to generate multiple transformed versions.

The results confirmed that the pipeline works correctly.

Example Results:

Original Image

Augmented Versions (Rotation, Shift, Zoom, Flip, Brightness changes, etc.)

(Insert screenshots from the notebook here to show original vs augmented images.)

4. Notes & Observations

A rotation range larger than ±20 caused distortions, so ±20 was chosen as optimal.

Augmentation is applied on-the-fly during training, meaning every epoch the model sees slightly different variations of the same image.

Validation data is not augmented; only training data is augmented to keep evaluation unbiased.

This approach increases dataset diversity without physically duplicating or storing extra images.

5. Files Created

src/augmentation/augmentation.py → Contains the augmentation pipeline function.

notebooks/augmentation_demo.ipynb → Used for testing and visualizing the augmentations.

reports/augmentation_report.md → Documentation of the augmentation process (this file).

6. Conclusion

The augmentation pipeline is ready to be integrated into the training process.
It significantly enhances dataset variability, improves model generalization, and helps prevent overfitting.
This stage completes the Data Augmentation milestone and prepares the dataset for the next stage: Model Training.