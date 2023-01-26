# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 17:50:01 2022

@author: pokhrel
"""

from layoutparser_pipeline import PDF_PreProcessing, OCR, PDF_PostProcessing
from transformers import pipeline
from transformers import FSMTForConditionalGeneration, FSMTTokenizer

def summarizer(df, column):
    classifier = pipeline("summarization", device='cpu')
    df["summarization"]=df[column].apply(lambda x: summarizer(x))
    return df
    
def question_answering(df, column, question):
    oracle = pipeline(model="deepset/roberta-base-squad2")
    df["answer_dict"] = df[column].apply(lambda x: oracle(question=question, context=x))
    return df
    
def translation_to_eng(df, column):
    mname = "facebook/wmt19-de-en"
    tokenizer = FSMTTokenizer.from_pretrained(mname)
    model = FSMTForConditionalGeneration.from_pretrained(mname)
    def apply_translator(sentence):
        input_ids = tokenizer.encode(sentence, return_tensors="pt")
        outputs = model.generate(input_ids)
        decoded_de_en = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded_de_en
    df["Translated_to_Eng"] = df[column].apply(apply_translator)
    return df
    
def translation_to_ger(df, column):
    mname = "facebook/wmt19-en-de"
    tokenizer = FSMTTokenizer.from_pretrained(mname)
    model = FSMTForConditionalGeneration.from_pretrained(mname)
    def apply_translator(sentence):
        input_ids = tokenizer.encode(sentence, return_tensors="pt")
        outputs = model.generate(input_ids)
        decoded_de_en = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded_de_en
    df["Translated_to_Eng"] = df[column].apply(apply_translator)
    return df

def keysentence_search(df, string, column, threshold):
    pass

def pipeline_nlp(pdf_file_path, output_path, model_architecture,
                extra_config, label_map, logger, task):
    
    #1. Preprocessing
    preprocesser = layoutparser_pipeline.PDF_PreProcessing( pdf_file_path=pdf_file_path,
                                                           output_path=output_path,
                                                           model_architecture=model_architecture,
                                                           extra_config=extra_config,
                                                           label_map=label_map,
                                                           logger=logger)
                                                           
    preprocessor.run()                                                           

    # Get the Layout Outputs
    layout_dict = preprocessor.get_Output()
    
    #2. OCR
    ocr_module = OCR(input_dict=layout_dict)
    ocr_module.run()
    ocr_dict = ocr_module.get_Output()
    
    #3. NLP
    
    #Convert ocr_dict to ocr_df
    

def main(_):
    pdf_file_path = None
    output_path = None
    
    model_architecture = None
    extra_config = None
    label_map = None
    logger = None
    
    task = None
    task_list = ["Summarization", "Question-Answering", "Translation", "Key Sentence Search"]
    
    if task in task_list:
        pipeline_nlp(pdf_file_path, output_path, model_architecture,
                extra_config, label_map, logger, task)
    

if __name__ == "__main__":
    main()
