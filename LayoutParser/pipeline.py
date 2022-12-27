# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 17:50:01 2022

@author: pokhrel
"""

import layoutparser_pipeline


def main(_):
    pdf_file_path = None
    output_path = None
    
    model_architecture = None
    extra_config = None
    label_map = None
    logger = None
    
    preprocesser = layoutparser_pipeline.PDF_PreProcessing(pdf_file_path=pdf_file_path,
                                                           output_path=output_path,
                                                           model_architecture=model_architecture,
                                                           extra_config=extra_config,
                                                           label_map=label_map,
                                                           logger=logger)
    # Load the model
    preprocesser.load_detectron()
    
    # Convert pdf to images
    preprocesser.convert_pdf_to_images()
    
    # Generate layouts
    preprocesser.scan_images()
    
    # Image blocks
    image_blocks = preprocesser.get_images_blocks()
    
    print(image_blocks)

if __name__ == "__main__":
    main()