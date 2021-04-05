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
