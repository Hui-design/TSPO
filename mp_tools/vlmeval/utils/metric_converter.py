from pandas import DataFrame


def convert(score: DataFrame, dataset_name: str):
    if dataset_name == "MME":
        return convert_mme_to_json(score)
    elif dataset_name == "CCBench":
        return convert_ccbench_to_json(score)
    elif dataset_name == "MMMU_DEV_VAL":
        return convert_mmmuval_to_json(score)
    elif dataset_name == "MMStar":
        return convert_mmstar_to_json(score)
    elif dataset_name == "MathVista":
        return convert_mathvista_to_json(score)
    elif "AI2D" in dataset_name:
        return convert_ai2d_to_json(score)
    elif "HallusionBench" in dataset_name:
        return convert_hallusionbench_to_json(score)
    elif "MMVet" in dataset_name:
        return convert_mmvet_to_json(score)


def convert_mmvet_to_json(score: DataFrame):
    result = {}
    category_mapping = {
        'rec': 'Rec',
        'ocr': 'Ocr',
        'know': 'Know',
        'gen': 'Gen',
        'spat': 'Spat',
        'math': 'Math',
        'Overall': 'Overall'
    }
    for _, row in score.iterrows():
        category = category_mapping.get(row['Category'], row['Category'])
        result[category] = round(row['acc'], 1)
    result['Overall (official)'] = 'N/A'
    return result


def convert_hallusionbench_to_json(score: DataFrame):
    result = {}
    overall_row = score[score['split'] == 'Overall'].iloc[0]
    result['aAcc'] = round(overall_row['aAcc'], 1)
    result['fAcc'] = round(overall_row['fAcc'], 1)
    result['qAcc'] = round(overall_row['qAcc'], 1)
    result['Overall'] = round(
        (result['aAcc'] + result['fAcc'] + result['qAcc']) / 3, 1)
    return result


def convert_ai2d_to_json(score: DataFrame):
    score_dict = score.iloc[0].to_dict()
    score_dict.pop("split")
    result = {key: round(score_dict[key] * 100, 1)
              for key in score_dict.keys()}
    return result


def convert_mathvista_to_json(score: DataFrame):
    result = {}
    task_skill_mapping = {
        "Overall": "Overall",
        "scientific reasoning": "SCI",
        "textbook question answering": "TQA",
        "numeric commonsense": "NUM",
        "arithmetic reasoning": "ARI",
        "visual question answering": "VQA",
        "geometry reasoning": "GEO",
        "algebraic reasoning": "ALG",
        "geometry problem solving": "GPS",
        "math word problem": "MWP",
        "logical reasoning": "LOG",
        "figure question answering": "FQA",
        "statistical reasoning": "STA"
    }

    for _, row in score.iterrows():
        task = row['Task&Skill']
        if task in task_skill_mapping:
            key = task_skill_mapping[task]
            result[key] = round(row['acc'], 1)

    return result


def convert_mmstar_to_json(score: DataFrame):
    score_dict = score.iloc[0].to_dict()
    score_dict.pop("split")
    result = {key: round(score_dict[key] * 100, 1)
              for key in score_dict.keys()}
    return result


def convert_mmmuval_to_json(score: DataFrame):
    score_dict = score[score['split'] == 'validation'].iloc[0].to_dict()
    keys = ("Overall", "Art & Design", "Business", "Science",
            "Health & Medicine", "Humanities & Social Science", "Tech & Engineering")
    result = {key: round(score_dict[key] * 100, 1) for key in keys}
    return result


def convert_mme_to_json(score: DataFrame):
    score_dict = score.iloc[0].to_dict()

    # Calculate overall score
    overall = score_dict['perception'] + score_dict['reasoning']

    # Create new dictionary with desired format
    result = {
        "Overall": round(overall, 1),
        "Perception": round(score_dict['perception'], 1),
        "Cognition": round(score_dict['reasoning'], 1),
        "OCR": round(score_dict['OCR'], 1),
        "Artwork": round(score_dict['artwork'], 1),
        "Celebrity": round(score_dict['celebrity'], 1),
        "Code Reasoning": round(score_dict['code_reasoning'], 1),
        "Color": round(score_dict['color'], 1),
        "Commonsense Reasoning": round(score_dict['commonsense_reasoning'], 1),
        "Count": round(score_dict['count'], 1),
        "Existence": round(score_dict['existence'], 1),
        "Landmark": round(score_dict['landmark'], 1),
        "Numerical Calculation": round(score_dict['numerical_calculation'], 1),
        "Position": round(score_dict['position'], 1),
        "Posters": round(score_dict['posters'], 1),
        "Scene": round(score_dict['scene'], 1),
        "Text Translation": round(score_dict['text_translation'], 1)
    }
    return result


def convert_ccbench_to_json(score: DataFrame):
    score_dict = score.iloc[0].to_dict()

    # Create new dictionary with desired format
    result = {
        "Overall": round(score_dict['Overall'] * 100, 1),
        "Calligraphy Painting": round(score_dict['calligraphy_painting'] * 100, 1),
        "Cultural Relic": round(score_dict['cultural_relic'] * 100, 1),
        "Food Clothes": round(score_dict['food_clothes'] * 100, 1),
        "Historical Figure": round(score_dict['historical_figure'] * 100, 1),
        "Scenery Building": round(score_dict['scenery_building'] * 100, 1),
        "Sketch Reasoning": round(score_dict['sketch_reasoning'] * 100, 1),
        "Traditional Show": round(score_dict['traditional_show'] * 100, 1)
    }
    return result
