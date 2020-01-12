# BJJ Position Classification
### Machine Learning classification for various positions in Brazilian Jiu Jitsu

To add a new classification to the model,
- Create a folder in the /Images subfolder, with the same name as the BJJ position
- Add images to the newly created folder, preferrably all with the same name with a trailing number for differentiation. JPG format is also preferred
- Add the name of the position/folder (which MUST be the same) to the "Categories" array in the Notebook
- Change the "Dense" variable to the new number of classifications

The end of the Notebook contains the cell where you can input your own image for the model to classify. Simply set the value of the "input_image" variable to the name of the file WITH the extension. 