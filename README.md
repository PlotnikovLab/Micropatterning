# Overview

Methods to control cell shape are commonly employed to understand signaling mechanisms and cytoskeletal dynamics in a cell. Although many techniques have been developed to manipulate cell shape, they often require access to specialized lithography equipment (e.g. clean rooms) or require the generation of photomasks, which is an extremely lengthy process. Our protocol generates micropatterns without the need of lithography equipment and photomasks, and is extremely flexible should patterns need to be changed on demand. All the researcher needs to do is import a new binary mask into an appropriate folder and the microscope will start patterning right away. This is especially convenient for early-stage experiments when the optimal shape and size of patterns have not yet been determined.

## Instructions

This repository contains two types of files - macros and .py scripts. The macros enable automated micropatterning on an infrared laser-equipped multiphoton microscope powered by NIS-Elements. The .py scripts allow users to analyze immunofluorescent images acquired from cells plated on such micropatterns.

### Macros

The macros contained in this repository serve to enable laser-assisted micropatterning using commercially available microscopes powered by NIS-Elements, especially those lacking hardware autofocus. Although the macros are designed for this specific software, the patterning workflow can be adapted and applied to different systems. 

* **pattern_stimulation**

	This macro runs through a series of optical configurations to ablate PVA in the current field of view (FOV). 
    
    First, a fiduciary marker is printed in the middle of the FOV through a *Z-stack ND experiment*. 
    ```
    SelectOptConf("Print_Fiduciary_Marker"); 
	ND_RunZSeriesExp();
    ```
    Next, the *Autofocus* function uses the boarders of the fiduciary marker to find the optimal z-plane. 
    ```
    SelectOptConf("Autofocus"); 
    StgFocus();
    ```
    Then, an image is acquired of this plane and the pattern mask is loaded onto the acquired image. The path of the ROI PatternMapPath should be defined in the stage_movement macro prior to running this macro.
    ```
    SelectOptConf("Load_ROI"); 
    CloseAllDocuments(2);
    Capture();
    ClearMeasROI();
    LoadROI(PatternMapPath,1);
    ```
    The regions of interest (ROIs) are set as stimulation ROIs. The ROI number (item preceding the comma) should be manually input before running the code. The following is only an example:
    ```
    ChangeROIType(1,3); 
    ChangeROIType(2,3);
    ChangeROIType(3,3);
    ChangeROIType(8,3);
    ChangeROIType(9,3);
    ChangeROIType(12,3);
    ChangeROIType(13,3);
    ChangeROIType(14,3);
    ```
    Finally, the microscope executes an *ND stimulation* experiment that repeatedly uses high laser power to stimulate the ROIs and remove PVA. It does so in a number of Z-planes to account for uneveness in the PVA layer and the stage so that maximum efficiency is achieved.
    ```
    A1ApplyStimulationSettings();
    SelectOptConf("Micropattern");
    ND_RunSequentialStimulationExp();
    ```
* **stage_movement**
	
    This macro allows the stage to move after each FOV has been patterned. To minimize Z-plane differences between sequential FOVs, the stage moves in a meander pattern.
    
    First, define the variables including pattern height, pattern lengt and step size.
    ```
    int PatternLength = 5; //In fields of view
	int PatternHeight = 5; //In fields of view
	double StepSize = 532.48; //In micrometers
    ```
    The path to the pattern_stimulation macro can be hardcoded or selected through a GUI, should a different pattern mask be used for each experiment.
    ```
    char PatternAblationMacroPath[256];
	global char PatternMapPath[256];
	int returned;
    ```

### .py scripts

The .py scripts in this repository enable averaging of immunofluorescent signals acquired from cells plated on micropatterns. Averaging such images can provide the researcher with a representative image that provides insight into the localization and abundance of proteins. 

The input should be .nd2 files with the same dimensions (number of pixels) and z slices with up to 4 channels. Place them in the same folder as the script .py file, the script will automatically identify all readable files in the folder and print the number of files found in the Console panel. Before running the script, please set parameters as indicated by the comments. Mandatory parameters include the channel to perform thresholding and whether optional features are enabled. For channel numbers, simply follow the orde that appear when viewing the files in other softwares (e.g. NIS Elements, Fiji ImageJ), from left (1) to right (4).

First, the script will perform maximum intensity projection for each image in order to integrate all features on different layers. Next, the script uses one of the channels, as determined by the user, to identify the shapes of cells and look for centroids. The result will be displayed in the Plots panel. If this step does not outline the cells properly, please adjust the offset in the parameters section by ~0.1. The images will then be aligned and averaged to create a representative image of all cells, split into up to four .tif files depending on the number of channels. An excel file containing the average intensities within each cell for each channel will also be generated for quantification purposes. Due to the large amount of data, we recommend cropping the images to remove unnecessary portions. The processing time can be a few minutes, and the progress can be tracked in the Console panel.

If optional features are enabled, the script will generate some additional results. The first group of .tif files are also averaged images, but each pixel is normalized to the average intensity of the corresponding channel within a cell. This avoids the result being dominated by few very bright images or features. For the last group of .tif files, the script uses a channel specified by the user to find the focal plane for each cell by comparing intensities of each slice. A file containing all focused slices of another channel of interest will be generated for convenience. Then, the script create another averaged image using only the focused slices as sources. A normalized version will also be created as described above. This feature is primarily designed for detecting membrane-localized signals that are close to the coverslip surface without interference from the cytoplasm or the volume of the cell.
