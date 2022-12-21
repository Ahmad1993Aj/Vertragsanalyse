# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 13:19:18 2022

@author: pokhrel
"""

import layoutparser as lp
import cv2
import os
import sys
import logging
import pdf2image


class PDF_PreProcessing:
    """
    """
    def __init__(self, pdf_file_path, output_path, 
                 model_architecture=None,
                 extra_config=None,
                 label_map = None,
                 logger=None):
        
        self.pdf_file_path = pdf_file_path
        self.output_path = output_path
        
        self.model = None
        self.model_architecture = model_architecture
        self.extra_config = extra_config
        self.label_map = label_map
        
        self.logger = logging.getLogger("pdfLogger") if logger==None else logger
        
    def scan_images(self):
        """
        This function creates layout of the images given a specific model and a
        label_map

        Returns
        -------
        None.

        """
        
        images = self.getImages()
        
        for image in images:
            pass
    
    def convert_pdf_to_images(self):
        """
        This function converts a given pdf file to set of jpg images and
        stores the jpeg images in the output_path

        Returns
        -------
        None.

        """
        # Check whether pdf file actually exists or not
        if not(os.path.exists(self.pdf_file_path)):
            self.logger.ERROR("PDF file in the given path {} does not exist..."\
                          .format(self.pdf_file_path))
            self.logger.INFO("Exiting the program...")
            sys.exit(1)
        
        # Create output dictionary if it does not exist
        if not(os.path.exists(self.output_path)):
            self.logger.INFO("Output path {} does not exist..."\
                          .format(self.output_path))
            os.mkdir(self.output_path)
            self.logger.INFO("Created the output directory {} ..."\
                             .format(self.output_path))

        images = pdf2image.convert_from_path(self.pdf_file_path)
        self.logger.INFO("Converted the images...")

        for i in range(len(images)):   
            # Save pages as images in the pdf
            images[i].save(self.output_path+'page'+ str(i) +'.jpg', 'JPEG')
        self.logger.INFO("Stored the images in the given path {} ...".format(self.output_path))
    
    def load_detectron(self):
        """
        This method loads the detectron model
        
        Returns
        -------
        None.

        """
        self.model = lp.Detectron2LayoutModel(self.model_architecture,
                                 extra_config=self.extra_config,
                                 label_map=self.label_map)
        
    def getImages(self):
        """
        

        Returns
        -------
        Images : List
            Contains list of the images
        """
        Images = []
        return Images

    
class PDF_PostProcessing:
    """
    """
    def __init__(self):
        pass
    
class OCR:
    """
    """
    def __init__(self):
        self.agent = None
    
    def load_agent(self):
        """
        Loads the ocr agent.

        Returns
        -------
        None.

        """
        self.agent = lp.GCVAgent.with_credential("<path/to/your/credential>",
                                       languages = ['en'])
        