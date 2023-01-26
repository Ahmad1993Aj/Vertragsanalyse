# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 13:19:18 2022

@author: pokhrel
"""

import layoutparser as lp
import cv2
import os
import re
import sys
import logging
import pdf2image
import pandas as pd


class PDF_PreProcessing:
    """
    """
    def __init__(self, pdf_file_path, output_path, 
                 model_architecture=None,
                 extra_config=None,
                 label_map=None,
                 logger=None,
                 labels_of_interest=["Text", "Title"]):
        
        self.pdf_file_path = pdf_file_path
        self.output_path = output_path
        
        self.model = None
        self.model_architecture = model_architecture
        self.extra_config = extra_config
        self.label_map = label_map
        self.loi = labels_of_interest
        
        self.images_outputs = dict()
        
        if logger==None:
            logging.basicConfig(filename=self.output_path+"logfile.txt",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

            logging.info("Layout Parser Pipeline")

            self.logger = logging.getLogger('pdfLogger')
        else:
            self.logger = logger
        
    def scan_images(self):
        """
        This function creates layout of the images given a specific model and a
        generates a dictionary with the follwoing information.
            *Coordinates:
            *text
            *id
            *type
            *parent
            *next
            *score
        Returns
        -------
        """
        
        images = self.getImages()
        result = {"image_path":[],
                  "page": [],
                  "page_elem":[],
                  "x_1":[],
                  "x_2":[],
                  "y_1":[],
                  "y_2":[],
                  "text":[],
                  "id":[],
                  "type":[],
                  "parent":[],
                  "next":[],
                  "score":[],
                  }
        
        for p, image_path in enumerate(images):
            print("Image path: {}".format(image_path))
            image = cv2.imread(image_path)
            image_layout = self.model.detect(image)
            filtered_layout = lp.Layout([b for b in image_layout if b.type in self.loi])
 
            for i, layout in enumerate(filtered_layout):
                
                result["image_path"].append(image_path)
                result["page"].append(p)
                result["page_elem"].append(i)
                result["x_1"].append(layout.block.x_1)
                result["x_2"].append(layout.block.x_2)
                result["y_1"].append(layout.block.y_1)
                result["y_2"].append(layout.block.y_2)
                result["text"].append(layout.text)
                result["id"].append(layout.id)
                result["type"].append(layout.type)
                result["parent"].append(layout.parent)
                result["next"].append(layout.next)
                result["score"].append(layout.score)
                result
                
        self.images_outputs = result

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
            self.logger.info("Exiting the program...")
            sys.exit(1)
        
        # Create output dictionary if it does not exist
        if not(os.path.exists(self.output_path)):
            self.logger.info("Output path {} does not exist..."\
                          .format(self.output_path))
            os.mkdir(self.output_path)
            self.logger.info("Created the output directory {} ..."\
                             .format(self.output_path))

        images = pdf2image.convert_from_path(self.pdf_file_path)
        self.logger.info("Converted the images...")

        for i in range(len(images)):   
            # Save pages as images in the pdf
            images[i].save(self.output_path+"page"+str(i) +'.jpg', 'JPEG')
        self.logger.info("Stored the images in the given path {} ...".format(self.output_path))
    
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
        self.logger.info("Loaded Detectron Model...")

    def getImages(self):
        """
        Returns
        -------
        Images : List
            Contains list of the images
        """
        Images = os.listdir(self.output_path)
        result = []
        for i, img in enumerate(Images):
            if img[-4:] == ".jpg":
                result.append(self.output_path + img)
        def natural_key(string_):
            """See https://blog.codinghorror.com/sorting-for-humans-natural-sort-order/"""
            return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]
        result = sorted(result, key=natural_key)
        return result
        
    def get_Output(self):
        """
        Returns
        -------
        Image Outputs : dict
            Contains list of the output of the layouts
        """
        return self.images_outputs
        
    def run(self):
        """
        This method is a simpe pipeline of the PreProcessing class.
        The steps are the following one:
        
        1. Convert PDF to a collection of Images
        2. Load the detectron model
        3. Generate layouts of the images
        4. Store the layouts as a csv file in the output path
        Returns
        -------
        """
        self.convert_pdf_to_images()
        self.load_detectron()
        self.scan_images()
    
class OCR:
    """
    """
    def __init__(self, input_dict):
        
        self.agent = None
        self.output_dict = input_dict
        
    def load_agent(self):
        """
        Loads the ocr agent.

        Returns
        -------
        None.

        """
        self.agent = lp.TesseractAgent(languages="eng")
        
    def run(self):
    	self.load_agent()
    	self.output_dict["content"] = []
    	for i, image_path in enumerate(self.output_dict["image_path"]):

            x_1 = float(self.output_dict["x_1"][i])
            x_2 = float(self.output_dict["x_2"][i])
            y_1 = float(self.output_dict["y_1"][i])
            y_2 = float(self.output_dict["y_2"][i])
            text = self.output_dict["text"][i]
            row_id = self.output_dict["id"][i]
            row_type = self.output_dict["type"][i]
            parent = self.output_dict["parent"][i]
            next = self.output_dict["next"][i]
            score = float(self.output_dict["score"][i])
            
            block = lp.TextBlock(block=lp.Rectangle(x_1=x_1, y_1=y_1, x_2=x_2, y_2=y_2), text=text, id=row_id, type=row_type, parent=parent, next=next, score=score)

            image = cv2.imread(image_path)
            
            segment_image = (block.pad(left=5, right=5, top=5, bottom=5).crop_image(image))
            
            text = self.agent.detect(segment_image)
            self.output_dict["content"].append(text)
            
    def get_Output(self):
        """
        Returns
        -------
        Output : dict
            Contains list of the output and the ocr content of the layouts
        """
        return self.output_dict

class PDF_PostProcessing:
    """
    """
    def __init__(self, output_option, save_dir, input_dict):
        self.output_option = output_option
        self.save_dir = save_dir
        self.output_dict = input_dict
        
    def get_OCR_Output(self):
        pass
        
    def get_NLP_Output(self):
        pass
        
    def get_Output(self):
        pass
        
    def run(self):
        pass
