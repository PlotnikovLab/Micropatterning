# Overview

Methods to control cell shape are commonly employed to understand signaling mechanisms and cytoskeletal dynamics in a cell. Although many techniques have been developed to manipulate cell shape, they often require access to specialized lithography equipment (e.g. clean rooms) or require the generation of photomasks, which is an extremely lengthy process. Our protocol generates micropatterns without the need of lithography equipment and photomasks, and is extremely flexible should patterns need to be changed on demand. All the researcher needs to do is import a new binary mask into an appropriate folder and the microscope will start patterning right away. This is especially convenient for early-stage experiments when the optimal shape and size of patterns have not yet been determined.

## Instructions

This repository contains two types of files - macros and .py scripts. The macros enable automated micropatterning on an infrared laser-equipped multiphoton microscope powered by NIS-Elements. The .py scripts allow users to analyze immunofluorescent images acquired from cells plated on such micropatterns.

### Macros

The macros contained in this repository serve to enable laser-assisted micropatterning using a commercially available microscope powered by NIS-Elements. Although the macros are designed for this specific software, the patterning workflow can be adapted and applied to different systems. 

### .py scripts

The .py scripts in this repository enable averaging of immunofluorescent signals acquired from cells plated on micropatterns. Averaging such images can provide the researcher with a representative image that provides insight into the localization and abundance of proteins. 
