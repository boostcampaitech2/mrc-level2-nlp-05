from transformers import EvalPrediction, TrainingArguments

from utils_qa import postprocess_qa_predictions

from arguments import DatasetArguments

# TODO: need to be improved!
def post_processing_function(
    examples, 
    features, 
    datasets,
    predictions, 
    training_args: TrainingArguments, 
    dataset_args: DatasetArguments
):

    column_names  = features.column_names
    answer_column = "answers" if "answers" in column_names else column_names[2]

    # Post-processing: start logits과 end logits을 original context의 정답과 match시킵니다.
    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        predictions=predictions,
        max_answer_length=dataset_args.max_ans_len,
        output_dir=training_args.output_dir,
    )
    # Metric을 구할 수 있도록 Format을 맞춰줍니다.
    formatted_predictions = [
        {"id": k, "prediction_text": v} for k, v in predictions.items()
    ]
    if training_args.do_predict:
        return formatted_predictions

    elif training_args.do_eval:
        references = [
            {"id": ex["id"], "answers": ex["answers"]}
            for ex in datasets["validation"]
        ]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

