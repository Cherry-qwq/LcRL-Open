# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the MKQA dataset to parquet format with multi-language support
"""

import pandas as pd
import os
import datasets
from sklearn.model_selection import train_test_split
import json
import argparse

def filter_null_answers(df: pd.DataFrame) -> pd.DataFrame:
    """
    过滤掉含有 null ground truth 的样本。
    判断标准与 count_null_answers 脚本一致：
    只要 answers 里某个语言的某个 answer 的 text 为 None，就认为这一行有 null answer。
    """

    def has_null_answer(answers):
        
        if not isinstance(answers, dict):
            return True  # 结构异常也直接当作无效样本

        for lang, answer_list in answers.items():
            if answer_list is None:
                return True
            for answer in answer_list:
                if answer.get("text") is None:
                    return True
        return False

    mask = ~df["answers"].apply(has_null_answer)
    kept = mask.sum()
    removed = len(df) - kept
    print(f"[Filter] Total rows: {len(df)}, kept: {kept}, removed (null ground truth): {removed}")
    return df[mask].reset_index(drop=True)

LANGUAGE_TO_SOURCE = {
        'fr': 'it',  
        'es': 'fr',   
        'pt': 'fr',  
        'it': 'fr',   
        'de': 'fr',   
        'en': 'fr',   
        'ru': 'fr',  
        'zh_cn': 'ja',
        'zh': 'ja',
        'ja': 'zh_cn',   
        'ko': 'ja',   
        'th': 'ja',   
        'fi': 'ru',  
        'ar': 'fr', 
    }
def make_prefix(dp, lang, template_type='base', question_lang=None):
    """
    生成多语言 prompt，支持多种模板类型
    """
    if question_lang is None:
        question_lang = lang
    question = dp.queries[question_lang]
    question = question.strip()
    
    # 确保问题以问号结尾
    if question[-1] not in ['?', '？', '؟']:  # 添加阿拉伯语问号
        question += '?'
    
    # 🔴 语言名称映射（用于 prompt 中显示）
    language_names = {
        'en': 'English',
        'fr': 'French',
        'es': 'Spanish',
        'de': 'German',
        'zh_cn': 'Chinese',
        'ja': 'Japanese',
        'ar': 'Arabic',
        'ru': 'Russian',
        'pt': 'Portuguese',
        'it': 'Italian',
        'ko': 'Korean',
        'fi': 'Finnish',
        'th': 'Thai',
    }
    language_to_high_resource = {
                # 罗曼语系（拉丁语系）- 选择西班牙语或法语作为辅助
                'fr': 'Italian',   
                'es': 'French',   
                'pt': 'French',   
                'it': 'French',   
                
                'de': 'French',   
                'en': 'French',   
                

                'ru': 'French',   
                
                # 汉藏语系 - 中文和日语
                'zh_cn': 'Japanese',   # 中文 -> 日语（汉字文化圈）
                'ja': 'Chinese',   # 日语 -> 中文（汉字文化圈）
 
                'ko': 'Japanese',   
                'th': 'Japanese',   
                

                'fi': 'Russian',   
                'ar': 'French',  
            }
    

    instr_lang = question_lang
    lang_name = language_names.get(instr_lang, 'English')
    high_resource_lang = language_to_high_resource.get(instr_lang, 'English')
    # 🔴 根据语言选择示例答案
    example_answers = {
        'en': 'Beijing',
        'fr': 'l’arctique',
        'es': 'El ártico',
        'de': 'arktis',
        'zh_cn': '北京',
        'ja': '東京',
        'ar': 'القاهرة',
        'ru': 'Москва',
        'pt': 'O pólo norte',
        'it': 'artico',
        'ko': '서울',
        'hi': 'दिल्ली',
        'tr': 'Ankara',
    }

    answer_lang_name = language_names.get(lang, 'English')
    example_answer = example_answers.get(lang, 'Answer')
    
    # 🔴 不同模板类型
    if template_type == 'base':
        """适用于所有语言的基础模板（英语指令）"""
        prefix = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. \
Note: Answers should be in {answer_lang_name}. For example, <answer> {example_answer} </answer>. 
Question: {question}"""
        
    elif template_type == 'multiingual':
        """适用于所有语言的基础模板（英语指令）"""
        prefix = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> and </search>, and it will return the top searched results between <information> and </information>. \
You can search at least 2 times: first search in {lang_name} (or a high resource language) knowledge base, and second search in both English AND another high resource languages like {high_resource_lang} to provide more comprehensive information. If needed, continue searching in English \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. \
Note: The answer should be in {answer_lang_name}. For example, <answer> {example_answer} </answer>. 

Question: {question}"""

    elif template_type == 'multilang_instruction':
        """多语言指令版本（根据语言自动调整指令语言）"""
        instructions = {
            'en': f"Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search>, and it will return the top searched results between <information> and </information>. You can search as many times as you want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> {example_answer} </answer>.",
            'zh_cn': f"请回答以下问题。每次获得新信息后，你必须在<think>和</think>之间进行推理。推理后，如果发现缺少某些知识，可以通过<search> 查询 </search>调用搜索引擎，搜索结果将在<information>和</information>之间返回。你可以根据需要多次搜索。如果不需要进一步的外部知识，可以直接在<answer>和</answer>之间提供答案，无需详细解释。例如：<answer> {example_answer} </answer>。",
            'fr': f"Répondez à la question donnée. Vous devez d'abord raisonner entre <think> et </think> chaque fois que vous obtenez de nouvelles informations. Après le raisonnement, si vous manquez de connaissances, vous pouvez appeler un moteur de recherche avec <search> requête </search>, et il renverra les meilleurs résultats entre <information> et </information>. Vous pouvez rechercher autant de fois que vous le souhaitez. Si vous n'avez plus besoin de connaissances externes, vous pouvez fournir directement la réponse entre <answer> et </answer>. Par exemple, <answer> {example_answer} </answer>.",
            'es': f"Responda a la pregunta dada. Primero debe razonar dentro de <think> y </think> cada vez que obtenga nueva información. Después de razonar, si encuentra que le falta conocimiento, puede llamar a un motor de búsqueda con <search> consulta </search>, y devolverá los mejores resultados entre <information> y </information>. Puede buscar tantas veces como desee. Si no necesita más conocimiento externo, puede proporcionar directamente la respuesta entre <answer> y </answer>. Por ejemplo, <answer> {example_answer} </answer>.",
            'de': f"Beantworten Sie die gegebene Frage. Sie müssen zunächst zwischen <think> und </think> argumentieren, jedes Mal wenn Sie neue Informationen erhalten. Nach dem Argumentieren, wenn Sie feststellen, dass Ihnen Wissen fehlt, können Sie eine Suchmaschine mit <search> Abfrage </search> aufrufen, und sie gibt die besten Ergebnisse zwischen <information> und </information> zurück. Sie können so oft suchen, wie Sie möchten. Wenn Sie kein weiteres externes Wissen benötigen, können Sie die Antwort direkt zwischen <answer> und </answer> angeben. Zum Beispiel, <answer> {example_answer} </answer>.",
            'ja': f"与えられた質問に答えてください。新しい情報を得るたびに、まず<think>と</think>の間で推論する必要があります。推論の後、知識が不足していることがわかった場合は、<search>クエリ</search>で検索エンジンを呼び出すことができ、<information>と</information>の間に検索結果が返されます。何度でも検索できます。さらなる外部知識が不要な場合は、<answer>と</answer>の間に直接答えを提供できます。例：<answer> {example_answer} </answer>。",
            'ar': f"أجب على السؤال المطروح. يجب عليك أولاً التفكير بين <think> و </think> في كل مرة تحصل فيها على معلومات جديدة. بعد التفكير، إذا وجدت أنك تفتقر إلى بعض المعرفة، يمكنك استدعاء محرك بحث بواسطة <search> استعلام </search>، وسيعيد أفضل النتائج بين <information> و </information>. يمكنك البحث عدة مرات كما تريد. إذا لم تكن بحاجة إلى مزيد من المعرفة الخارجية، يمكنك تقديم الإجابة مباشرة بين <answer> و </answer>. على سبيل المثال، <answer> {example_answer} </answer>.",
            'ru': f"Ответьте на заданный вопрос. Вы должны сначала рассуждать между <think> и </think> каждый раз, когда получаете новую информацию. После рассуждения, если вы обнаружите, что вам не хватает знаний, вы можете вызвать поисковую систему с помощью <search> запрос </search>, и она вернет лучшие результаты между <information> и </information>. Вы можете искать столько раз, сколько захотите. Если вам не нужны дополнительные внешние знания, вы можете напрямую предоставить ответ между <answer> и </answer>. Например, <answer> {example_answer} </answer>.",
            'pt': f"Responda à pergunta dada. Você deve primeiro raciocinar entre <think> e </think> cada vez que obtiver novas informações. Após o raciocínio, se você descobrir que lhe falta conhecimento, pode chamar um mecanismo de pesquisa com <search> consulta </search>, e ele retornará os melhores resultados entre <information> e </information>. Você pode pesquisar quantas vezes quiser. Se você não precisar de mais conhecimento externo, pode fornecer diretamente a resposta entre <answer> e </answer>. Por exemplo, <answer> {example_answer} </answer>.",
            'it': f"Rispondi alla domanda data. Devi prima ragionare tra <think> e </think> ogni volta che ottieni nuove informazioni. Dopo aver ragionato, se scopri di mancare di conoscenze, puoi chiamare un motore di ricerca con <search> query </search>, e restituirà i migliori risultati tra <information> e </information>. Puoi cercare tutte le volte che vuoi. Se non hai bisogno di ulteriori conoscenze esterne, puoi fornire direttamente la risposta tra <answer> e </answer>. Ad esempio, <answer> {example_answer} </answer>.",
            'ko': f"주어진 질문에 답하십시오. 새로운 정보를 얻을 때마다 먼저 <think>와 </think> 사이에서 추론해야 합니다. 추론 후 지식이 부족하다는 것을 발견하면 <search> 쿼리 </search>로 검색 엔진을 호출할 수 있으며, <information>과 </information> 사이에 상위 검색 결과를 반환합니다. 원하는 만큼 여러 번 검색할 수 있습니다. 추가 외부 지식이 필요하지 않으면 <answer>와 </answer> 사이에 직접 답변을 제공할 수 있습니다. 예: <answer> {example_answer} </answer>.",
            'hi': f"दिए गए प्रश्न का उत्तर दें। जब भी आपको नई जानकारी मिले, आपको पहले <think> और </think> के बीच तर्क करना चाहिए। तर्क के बाद, यदि आपको लगता है कि आपके पास कुछ ज्ञान की कमी है, तो आप <search> क्वेरी </search> के साथ खोज इंजन को कॉल कर सकते हैं, और यह <information> और </information> के बीच शीर्ष खोज परिणाम लौटाएगा। आप जितनी बार चाहें खोज सकते हैं। यदि आपको और बाहरी ज्ञान की आवश्यकता नहीं है, तो आप <answer> और </answer> के बीच सीधे उत्तर प्रदान कर सकते हैं। उदाहरण के लिए, <answer> {example_answer} </answer>.",
            'tr': f"Verilen soruyu cevaplayın. Her yeni bilgi aldığınızda önce <think> ve </think> arasında akıl yürütmelisiniz. Akıl yürüttükten sonra, bilgi eksikliği olduğunu fark ederseniz, <search> sorgu </search> ile bir arama motoru çağırabilirsiniz ve <information> ve </information> arasında en iyi arama sonuçlarını döndürecektir. İstediğiniz kadar arama yapabilirsiniz. Daha fazla harici bilgiye ihtiyacınız yoksa, <answer> ve </answer> arasında doğrudan cevabı verebilirsiniz. Örneğin, <answer> {example_answer} </answer>.",
        }
        
        instruction = instructions.get(lang, instructions['en'])
        prefix = f"{instruction}\n\nQuestion: {question}"
    
    else:
        raise NotImplementedError(f"Template type '{template_type}' not implemented")
    
    return prefix


def create_dataset(df, language, split, template_type='base'):
    data = []
    
    for idx, row in enumerate(df.itertuples(), start=0):
        target_answers = []
        for answer_item in row.answers[language]:
            if 'text' in answer_item:
                target_answers.append(answer_item['text'])
            if 'aliases' in answer_item:
                target_answers.extend(answer_item['aliases'])
        question_prompt = make_prefix(row, language, template_type=template_type)

        ###构造content_en和content_source
        content_en = None
        if isinstance(row.queries, dict) and 'en' in row.queries and row.queries['en'] is not None:
            try:
                # 直接复用同一个模板逻辑，只是 lang = 'en'
                content_en = make_prefix(row, lang=language, template_type=template_type, question_lang='en' )
            except Exception as e:
                print(f"[WARN] Failed to build content_en for idx={idx}, language={language}: {e}")
                content_en = None
        content_source = None
        source_lang = LANGUAGE_TO_SOURCE.get(language)
        if source_lang is not None:
            if isinstance(row.queries, dict) and source_lang in row.queries and row.queries[source_lang] is not None:
                try:
                    content_source = make_prefix(row, lang=language, template_type=template_type, question_lang=source_lang)
                except Exception as e:
                    print(f"[WARN] Failed to build content_source for idx={idx}, language={language}, source_lang={source_lang}: {e}")
                    content_source = None
            else:
                print(f"[INFO] No query found for source_lang={source_lang} when processing language={language}, idx={idx}")

        solution = {
            "target": target_answers,
        }
        
        example = {
            "id": f"{language}_{split}_{idx}",
            "question": row.queries[language].strip(),
            "golden_answers": row.answers[language],
            "language": language,  
            "data_source": "mkqa",
            "prompt": [{
                "role": "user",
                "content": question_prompt,
                # "content_en": content_en,           # 并行英文版本
                # "content_source": content_source,
            }],
            "ability": "fact-reasoning",
            "reward_model": {
                "style": "rule",
                "ground_truth": solution
            },
            "extra_info": {
                'split': split,
                'index': idx,
                'language': language,  
            },
            "example_id":row.example_id
        }
        
        data.append(example)
    
    return data


def process_language(file_path, language, template_type, local_dir, hdfs_dir=None):
    """
    处理单个语言的数据：
    1. 读取 jsonl 为 DataFrame
    2. 过滤掉 ground truth 含 null 的样本
    3. 从剩余样本中随机抽取 2000 条作为 test（不够就全做 test）
       其余全部作为 train
    """
    print(f"\n{'='*60}")
    print(f"Processing language: {language.upper()}")
    print(f"{'='*60}")
    
    df = pd.read_json(file_path, lines=True)
    print(f"[Raw] Total rows before filtering: {len(df)}")
    
    df = filter_null_answers(df)
    print(f"[Filtered] Total rows after filtering: {len(df)}")
    
    if len(df) == 0:
        print(f"[WARNING] No valid rows left for language {language} after filtering. Skip.")
        return

    num_test = min(2000, len(df))
    print(f"[Split] Will use {num_test} examples as TEST, {len(df) - num_test} as TRAIN.")
    
    test_df = df.sample(n=num_test, random_state=42)
    train_df = df.drop(test_df.index)

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    
    train_data = create_dataset(train_df, language, 'train', template_type)
    test_data = create_dataset(test_df, language, 'test', template_type)
    
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    
    train_file = os.path.join(local_dir, f'train_{language}.parquet')
    test_file = os.path.join(local_dir, f'test_{language}.parquet')
    
    # 如果样本不足 2000，train_data 可能是空的，这里也安全保存
    pd.DataFrame(train_data).to_parquet(train_file)
    pd.DataFrame(test_data).to_parquet(test_file)
    
    print(f"✓ Saved: {train_file}")
    print(f"✓ Saved: {test_file}")
    
    # HDFS 存储（如果需要）
    if hdfs_dir is not None:
        from verl.utils.hdfs_io import copy, makedirs
        makedirs(hdfs_dir)
        copy(src=train_file, dst=os.path.join(hdfs_dir, f'train_{language}.parquet'))
        copy(src=test_file, dst=os.path.join(hdfs_dir, f'test_{language}.parquet'))
        print(f"✓ Uploaded to HDFS: {hdfs_dir}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/mkqa/cross', help='Local directory to save parquet files')
    parser.add_argument('--hdfs_dir', default=None, help='HDFS directory (optional)')
    parser.add_argument('--template_type', type=str, default='base', 
                       choices=['base', 'multilang_instruction', 'base-hr'],
                       help='Template type for prompt generation')
    parser.add_argument('--languages', type=str, default='all',
                       help='Comma-separated language codes (e.g., "en,fr,es") or "all" for all languages')
    parser.add_argument('--data_file', type=str, 
                       default='/local_data/ruiqi/Search-R1/data/mkqa/mkqa.jsonl',
                       help='Path to MKQA jsonl file')

    args = parser.parse_args()
    
    #定义支持的所有语言
    all_languages = ['en', 'fr', 'es', 'de', 'zh_cn', 'ja', 'ar', 'ru', 'pt', 'it', 'ko', 'fi', 'th']
    
    # 解析要处理的语言
    if args.languages == 'all':
        languages_to_process = all_languages
    else:
        languages_to_process = [lang.strip() for lang in args.languages.split(',')]
        invalid_langs = [lang for lang in languages_to_process if lang not in all_languages]
        if invalid_langs:
            print(f"[WARNING] Invalid language codes: {invalid_langs}")
            print(f"Supported languages: {all_languages}")
            languages_to_process = [lang for lang in languages_to_process if lang in all_languages]
    
    if not languages_to_process:
        print("[ERROR] No valid languages to process!")
        exit(1)
    
    print(f"\n{'='*60}")
    print(f"Will process {len(languages_to_process)} languages:")
    print(f"{', '.join(languages_to_process)}")
    print(f"Template type: {args.template_type}")
    print(f"{'='*60}")
    
    # 处理每种语言
    for language in languages_to_process:
        try:
            process_language(
                file_path=args.data_file,
                language=language,
                template_type=args.template_type,
                local_dir=args.local_dir,
                hdfs_dir=args.hdfs_dir
            )
        except Exception as e:
            print(f"[ERROR] Failed to process language {language}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("✓ All languages processed successfully!")
    print(f"Output directory: {args.local_dir}")
    print(f"{'='*60}\n")