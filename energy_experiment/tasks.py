"""Task definitions, text corpora, and prompt construction."""

GRAMMAR_ESSAY = """
The artificial intelligence have been change the way we lives and work in the past few year. Many companys are now using AI to automates they operations, but the impact on energy consumption are often overlooked. This is a serious problem. The weather was nice yesterday.

Large language models, which is a type of AI, requires massive amount of computational resources for to run. For example, training a single large model can consumed as much electricity as five homes uses in a entire year. However the inference phase which is when the model actually generate responses also use significant energy. People like to eat pizza on fridays.

The environmental implication of this energy usage is profound. Data centers which houses the servers that run these model account for approximately 1-2 percent of global electricity consumptions. This number are expected to growing significantly as AI adoption increase. The cat sat on the mat and looked at the window. Furthermore, the cooling system required to keep this servers running adds additional energy overhead that is often not accounted for in standard measurement.

Researchers has been studying various approach to reduce the energy footprint of AI system. One promising method are model quantization, which reduce the precision of the model parameter from 32-bit floating point to lower bit representation. This can reduces energy consumption by up to 50 percent while maintaining acceptable performance. However, there is trade-off between efficiency and quality. My favorite color is blue and I enjoy swimming.

Another approach that researcher have explored is knowledge distillation, where a smaller model are trained to mimics the behavior of a larger one. The smaller model can then be deployed in production, using significantly less energy per inference. This technique have shown promise in various application, from natural language processing to computer vision tasks. The restaurant down the street serves excellent pasta.

Batch processing is also a important factor in energy efficiency. When a system process multiple request simultaneously, the fixed cost of loading the model into memory are amortized across all request. This can lead to substantial energy saving per individual prompt. However in real-world deployment, latency requirement often limit the batch size that can be used, creating tension between energy efficiency and user experience. Birds fly south in the winter because they dont like cold weather.

The hardware on which models runs also play a crucial role in determine energy consumption. Different GPU architecture have varying level of energy efficiency. For example, newer GPU like the NVIDIA H100 can process more computation per watt compared to older model like the V100. But the total power draw of these newer chip are also significantly higher, meaning that the net energy impact depend on the specific workload and how efficiently the hardware are utilized. The book on the shelf was very interesting to read last summer.

Memory bandwidth is another critical factor that affect energy usage. During inference, large language model need to frequent access they weights from memory. If the memory bandwidth is insufficient, the GPU must wait for data, leading to underutilization and wasted energy. This is particularly problematic for larger model that cannot fit entirely in the GPU high-bandwidth memory. Solar panels is becoming more popular in many country around the world.

The relationship between model size and energy consumption is not straightforward. While larger model generally consume more energy, the relationship are not linear. A model with twice as many parameter does not necessarily consume twice as much energy. Factor such as the number of layer, the hidden dimension size, and the attention mechanism all play important role in determining the final energy consumption. The traffic was very bad this morning on my way to work.

Quantization method vary in there effectiveness. AWQ and GPTQ are two popular post-training quantization technique that can significant reduce model size and energy consumption. INT4 quantization, for example, can reduce the memory footprint by approximately 4 times compared to FP16, enabling larger model to run on smaller hardware. But the quality degradation must be carefully evaluate for each use case. I think that swimming is a good exercise for maintaining fitness.

The concept of energy per token is a useful metric for comparing different model and configuration. By measuring the total energy consumed during inference and divide it by the number of token generated, we can obtain a standardized measure of efficiency. This metric allow us to make fair comparison between model of different size running on different hardware. The sunset was beautiful yesterday evening over the mountain.

Looking at forward, several trend are likely to shapes the energy landscape of AI. First, more efficient hardware architecture, such as specialized AI accelerator and neuromorphic chip, may significantly reduce the energy per computation. Second, algorithm improvement, including more efficient attention mechanism and sparse model, could reduce the computational requirement of inference. Third, the grow adoption of renewable energy source for data center could reduce the carbon footprint even if total energy consumption continue to rise. My neighbor has a very cute dog that likes to play in the park every morning.

The development of new cooling technology is also important for reducing the overall energy footprint of AI system. Traditional air cooling are becoming insufficient for the high power density of modern GPU cluster. Liquid cooling solution, including direct-to-chip and immersion cooling, can significant improve cooling efficiency and reduce the energy overhead associated with thermal management. I really enjoy watching movies on rainy weekends with popcorn.

Edge computing present another opportunity for energy optimization. By deploying smaller, optimized model closer to the end user, the energy cost of data transmission can be reduced. This approach also reduce latency, which can improve the user experience. However edge device typically have more limited computational resource, which constrain the size and complexity of model that can be deployed. My grandmother makes the best apple pie in the whole neighborhood.

The role of software optimization in reduce energy consumption should not be underestimated. Efficient inference engine like vLLM and TensorRT-LLM can significant improve throughput and reduce energy usage through technique such as continuous batching, paged attention, and kernel fusion. These optimization operate at the system level and can provide benefit across different model architecture and hardware configuration. The new coffee shop downtown have really good lattes.

Transfer learning and fine-tuning are also relevant to the energy discussion. Rather than training large model from scratch for every application, researcher and practitioner can fine-tune existing pre-trained model on specific task. This approach dramatically reduce the computational cost and energy consumption associated with model development. Parameter-efficient fine-tuning method like LoRA further reduce the resource requirement by only updating a small subset of model parameter. Yesterday I saw a rainbow after the rain storm.

The measurement methodology for AI energy consumption need further standardization. Currently different study use different measurement approach, making it difficult to compare result across research group. Some study measure only GPU power, while other attempt to capture total system energy including CPU, memory, and cooling. Establishing standard benchmark and measurement protocol would greatly benefit the field. I wonder if aliens exist somewhere in the universe.

Carbon accounting for AI system is another area that require more attention. The carbon footprint of AI depend not only on energy consumption but also on the carbon intensity of the electricity source. A model running on a server powered by renewable energy will have a much lower carbon footprint than the same model running on coal-powered electricity. This highlight the importance of considering both energy efficiency and energy source in sustainability assessment. Pineapple on pizza is a controversial topic among food enthusiasts.

The economic implication of AI energy consumption are also significant. As energy cost rise, the operational expense of running large language model can become a major factor in business decision. Company must balance the benefit of using larger, more capable model against the increased energy cost. This economic pressure may drive adoption of more energy-efficient architecture and deployment strategy. My cat likes to sleep on the keyboard when I am trying to work.

Federated learning offer a interesting approach to reduce centralized energy consumption. By training model across distributed device without centralizing data, federated learning can reduce the energy required for data transmission and centralized computation. However this approach introduce new challenge related to communication overhead and model convergence that can offset some of the energy saving. The old oak tree in the park must be at least a hundred years old.

The impact of model architecture on energy consumption extend beyond simple parameter count. Attention mechanism, which are central to transformer model, have quadratic complexity with respect to sequence length. This mean that processing longer input require disproportionately more energy. Recent innovation such as flash attention, linear attention, and sliding window attention aim to reduce this computational overhead while maintaining model quality. I forgot to buy milk at the grocery store yesterday afternoon.

Sparse model and mixture of expert architecture represent another promising direction for energy efficiency. By activating only a subset of model parameter for each input, these architecture can achieve the performance of much larger dense model while using significantly less computation per inference. However they introduce complexity in load balancing and routing that must be carefully managed. The marathon runner finished the race in under three hours.

The lifecycle energy cost of AI system extend beyond just training and inference. The manufacturing of GPU and other specialized hardware require significant energy and rare material. Additionally the embodied carbon in data center infrastructure including building, networking equipment, and storage system contribute to the overall environmental impact. A comprehensive assessment of AI sustainability should account for all these factor. My favorite season is autumn because of the beautiful changing leaves.

Policy and regulation may play a increasing role in addressing AI energy consumption. Some researcher have proposed energy labeling for AI model, similar to energy efficiency rating for appliance. This would provide transparency to consumer and business about the environmental cost of using different AI service. Government could also incentivize the development and deployment of energy-efficient AI through tax credit or regulatory requirement. The children was playing happily in the playground after school.

In conclusion, the energy consumption of large language model inference is a multifaceted challenge that require attention from researcher, practitioner, and policy maker alike. By systematic measuring and analyzing energy usage across different model, hardware, task type, and deployment configuration, we can identify opportunity for optimization and guide the development of more sustainable AI system. The journey toward energy-efficient AI is not just a technical challenge but also a social and economic imperative that will shapes the future of technology and its relationship with our planet. The flowers in my garden is blooming beautifully this spring season after all the rain we had.
"""

SPANISH_TEXT = """
La inteligencia artificial ha transformado profundamente la manera en que vivimos y trabajamos en los ultimos anios. Numerosas empresas estan adoptando esta tecnologia para automatizar sus operaciones, mejorar la eficiencia y reducir los costos operativos. Sin embargo, el impacto que tiene el consumo de energia de estos sistemas frecuentemente se pasa por alto en las discusiones publicas sobre el tema.

Los modelos de lenguaje de gran tamano, conocidos comunmente como LLMs por sus siglas en ingles, requieren cantidades enormes de recursos computacionales para funcionar correctamente. Por ejemplo, el entrenamiento de un solo modelo grande puede consumir tanta electricidad como la que utilizan cinco hogares durante un ano entero. Ademas, la fase de inferencia, que es cuando el modelo realmente genera respuestas para los usuarios, tambien consume una cantidad significativa de energia que no debe subestimarse.

Las implicaciones ambientales de este consumo energetico son muy profundas y merecen una atencion especial. Los centros de datos que albergan los servidores necesarios para ejecutar estos modelos representan aproximadamente entre el uno y el dos por ciento del consumo mundial de electricidad. Se espera que este numero crezca significativamente a medida que la adopcion de la inteligencia artificial se acelere en los proximos anios.

Los investigadores han estado estudiando diversas estrategias para reducir la huella energetica de los sistemas de inteligencia artificial. Un metodo prometedor es la cuantizacion de modelos, que reduce la precision de los parametros del modelo desde representaciones de punto flotante de 32 bits a representaciones de menor cantidad de bits. Esta tecnica puede reducir el consumo de energia hasta en un cincuenta por ciento mientras se mantiene un rendimiento aceptable para la mayoria de las aplicaciones practicas.

Otro enfoque que los investigadores han explorado con resultados interesantes es la destilacion de conocimiento. En este proceso, un modelo mas pequeno se entrena para imitar el comportamiento de uno mas grande. El modelo resultante puede desplegarse en produccion utilizando significativamente menos energia por cada inferencia. Esta tecnica ha demostrado ser prometedora en diversas aplicaciones, desde el procesamiento del lenguaje natural hasta las tareas de vision por computadora.

El procesamiento por lotes constituye tambien un factor muy importante en la eficiencia energetica de estos sistemas. Cuando un sistema procesa multiples solicitudes de manera simultanea, el costo fijo de cargar el modelo en la memoria se distribuye entre todas las solicitudes del lote. Esto puede resultar en ahorros sustanciales de energia por cada consulta individual. No obstante, en las implementaciones del mundo real, los requisitos de latencia frecuentemente limitan el tamano del lote que puede utilizarse de manera practica.

El hardware sobre el cual se ejecutan los modelos juega un papel fundamental en la determinacion del consumo energetico total. Las diferentes arquitecturas de unidades de procesamiento grafico tienen niveles variados de eficiencia energetica. Por ejemplo, las GPU mas recientes como la NVIDIA H100 pueden procesar mas operaciones por vatio en comparacion con modelos anteriores como la V100. Sin embargo, el consumo total de energia de estos chips mas nuevos es tambien considerablemente mayor.

El ancho de banda de la memoria es otro factor critico que afecta directamente el uso de energia durante la inferencia. Los modelos de lenguaje grandes necesitan acceder frecuentemente a sus pesos almacenados en la memoria. Si el ancho de banda de la memoria resulta insuficiente, la unidad de procesamiento grafico debe esperar los datos, lo que conduce a una subutilizacion del hardware y un desperdicio considerable de energia electrica.

La relacion entre el tamano del modelo y su consumo de energia no es directa ni sencilla de predecir. Aunque los modelos mas grandes generalmente consumen mas energia, esta relacion no es lineal en absoluto. Un modelo con el doble de parametros no necesariamente consume el doble de energia. Factores como el numero de capas, el tamano de la dimension oculta y el mecanismo de atencion desempenan un papel igualmente importante en la determinacion del consumo final.

Los metodos de cuantizacion varian considerablemente en su efectividad segun el caso de uso. AWQ y GPTQ son dos tecnicas populares de cuantizacion posterior al entrenamiento que pueden reducir significativamente tanto el tamano del modelo como su consumo de energia. La cuantizacion a INT4, por ejemplo, puede reducir la huella de memoria aproximadamente cuatro veces en comparacion con FP16, permitiendo que modelos mas grandes se ejecuten en hardware mas modesto.

El concepto de energia por token resulta ser una metrica muy util para comparar diferentes modelos y configuraciones de manera justa. Al medir la energia total consumida durante la inferencia y dividirla por el numero de tokens generados, podemos obtener una medida estandarizada de la eficiencia. Esta metrica nos permite realizar comparaciones justas entre modelos de diferentes tamanos que se ejecutan en hardware diferente.

Mirando hacia el futuro, varias tendencias probablemente moldearan el panorama energetico de la inteligencia artificial. En primer lugar, arquitecturas de hardware mas eficientes, como aceleradores especializados de inteligencia artificial y chips neuromorficos, podrian reducir significativamente la energia necesaria por cada operacion computacional. En segundo lugar, las mejoras algoritmicas podrian disminuir los requisitos computacionales de la inferencia.

El desarrollo de nuevas tecnologias de refrigeracion tambien es fundamental para reducir la huella energetica general de los sistemas de inteligencia artificial. Los sistemas tradicionales de refrigeracion por aire se estan volviendo insuficientes para la alta densidad de potencia de los clusters modernos de GPU. Las soluciones de refrigeracion liquida, incluyendo la refrigeracion directa al chip y la refrigeracion por inmersion, pueden mejorar significativamente la eficiencia termica.

La computacion en el borde presenta otra oportunidad interesante para la optimizacion energetica. Al desplegar modelos mas pequenos y optimizados mas cerca del usuario final, se puede reducir el costo energetico de la transmision de datos. Este enfoque tambien reduce la latencia, lo que puede mejorar considerablemente la experiencia del usuario. Sin embargo, los dispositivos en el borde tipicamente tienen recursos computacionales mas limitados.

El papel de la optimizacion del software en la reduccion del consumo energetico no debe subestimarse bajo ninguna circunstancia. Los motores de inferencia eficientes como vLLM y TensorRT-LLM pueden mejorar significativamente el rendimiento y reducir el uso de energia mediante tecnicas como el procesamiento continuo por lotes, la atencion paginada y la fusion de kernels computacionales.

El aprendizaje por transferencia y el ajuste fino tambien son relevantes para la discusion sobre la energia en el contexto de la inteligencia artificial. En lugar de entrenar modelos grandes desde cero para cada aplicacion especifica, los investigadores y profesionales pueden ajustar modelos preentrenados existentes para tareas concretas. Los metodos de ajuste fino eficientes en parametros como LoRA reducen aun mas los requisitos de recursos.

La metodologia de medicion del consumo energetico de la inteligencia artificial necesita una mayor estandarizacion a nivel global. Actualmente, diferentes estudios utilizan enfoques de medicion distintos, lo que dificulta enormemente la comparacion de resultados entre diferentes grupos de investigacion. Algunos estudios miden unicamente la potencia de la GPU, mientras que otros intentan capturar la energia total del sistema.

La contabilidad del carbono para los sistemas de inteligencia artificial es otra area que requiere mucha mas atencion de la que recibe actualmente. La huella de carbono de la inteligencia artificial depende no solo del consumo de energia sino tambien de la intensidad de carbono de la fuente de electricidad utilizada. Un modelo que se ejecuta en un servidor alimentado por energia renovable tendra una huella de carbono mucho menor.

Las implicaciones economicas del consumo energetico de la inteligencia artificial son tambien muy significativas y afectan las decisiones empresariales. A medida que los costos de la energia aumentan, los gastos operativos de ejecutar modelos de lenguaje grandes pueden convertirse en un factor determinante en las decisiones comerciales. Las empresas deben equilibrar los beneficios de utilizar modelos mas grandes y capaces contra el incremento en los costos energeticos.

El aprendizaje federado ofrece un enfoque interesante para reducir el consumo de energia centralizado en los centros de datos. Al entrenar modelos a traves de dispositivos distribuidos sin centralizar los datos del usuario, el aprendizaje federado puede reducir la energia requerida para la transmision de datos y la computacion centralizada. Sin embargo, este enfoque introduce nuevos desafios relacionados con la sobrecarga de comunicacion entre dispositivos.

El impacto de la arquitectura del modelo en el consumo de energia se extiende mucho mas alla del simple conteo de parametros. Los mecanismos de atencion, que son fundamentales para los modelos transformer modernos, tienen una complejidad cuadratica con respecto a la longitud de la secuencia de entrada. Esto significa que procesar entradas mas largas requiere una cantidad desproporcionadamente mayor de energia computacional.

En conclusion, el consumo de energia de la inferencia de modelos de lenguaje grandes es un desafio multifacetico que requiere la atencion coordinada de investigadores, profesionales y responsables de politicas por igual. Mediante la medicion y el analisis sistematicos del uso de energia en diferentes modelos, hardware, tipos de tareas y configuraciones de despliegue, podemos identificar oportunidades concretas para la optimizacion y guiar el desarrollo de sistemas de inteligencia artificial mas sostenibles y responsables con el medio ambiente para las generaciones futuras.
"""

# ── Task types ──
# 'text_processing': has source_text that gets sliced to target_tokens sizes
# 'generation': has fixed prompts (short questions), input is constant

TASKS = {
    # ── Text processing tasks (variable input length) ──
    'grammar_fix': {
        'type': 'text_processing',
        'description': 'Fix grammar, spelling, and cohesion errors in English text',
        'source_text': GRAMMAR_ESSAY,
        'prompt_template': (
            "The following text contains grammar mistakes, spelling errors, and "
            "sentences that break the flow or are completely unrelated to the topic. "
            "Rewrite the text fixing all grammar and spelling errors, and remove or "
            "replace any sentences that are off-topic or break cohesion. "
            "Return only the corrected text, nothing else.\n\n"
            "TEXT TO FIX:\n{input_text}"
        ),
    },
    'translation': {
        'type': 'text_processing',
        'description': 'Translate Spanish text to clear, natural English',
        'source_text': SPANISH_TEXT,
        'prompt_template': (
            "Translate the following Spanish text into clear, natural English. "
            "Maintain the original meaning, tone, and paragraph structure. "
            "Return only the English translation, nothing else.\n\n"
            "SPANISH TEXT:\n{input_text}"
        ),
    },

    # ── Content generation tasks (fixed short input, variable output) ──
    'content_generation': {
        'type': 'generation',
        'description': 'Generate essay/article content from a short question',
        'prompts': [
            {
                'id': 'essay_pollution',
                'text': 'Write an essay on environment pollution.',
            },
            {
                'id': 'explain_cancer',
                'text': 'What is colon cancer? Explain.',
            },
            {
                'id': 'explain_food',
                'text': 'What is the healthiest food? Explain.',
            },
        ],
    },

    # ── Code generation tasks (fixed short input, fixed output) ──
    'code_generation': {
        'type': 'generation',
        'description': 'Generate Python code from a short specification',
        'prompts': [
            {
                'id': 'csv_sum',
                'text': (
                    'Write Python code to read a CSV file and take the summation '
                    'of the total column. Return only the code, nothing else.'
                ),
            },
            {
                'id': 'reverse_array',
                'text': (
                    'Write Python code to reverse iterate an array. '
                    'Return only the code, nothing else.'
                ),
            },
        ],
    },
}


def get_task_prompts(task):
    """Get the list of prompts for a generation task.

    Returns list of dicts with 'id' and 'text' keys.
    For text_processing tasks, returns None (use get_task_input instead).
    """
    if task not in TASKS:
        raise ValueError(f'Unknown task: {task}. Available: {list(TASKS.keys())}')

    info = TASKS[task]
    if info.get('type') == 'generation':
        return info['prompts']
    return None


def get_task_input(task, target_tokens):
    """Get input text for a task.

    For text_processing tasks: returns source text sliced to target_tokens.
    For generation tasks: returns the prompt at index (target_tokens % num_prompts).
    """
    if task not in TASKS:
        raise ValueError(f'Unknown task: {task}. Available: {list(TASKS.keys())}')

    info = TASKS[task]

    if info.get('type') == 'generation':
        prompts = info['prompts']
        idx = target_tokens % len(prompts)
        return prompts[idx]['text']

    # text_processing
    target_chars = target_tokens * 4
    source = info['source_text'].strip()

    full_text = source
    while len(full_text) < target_chars:
        full_text += '\n\n' + source

    return full_text[:target_chars]


def build_prompt(task, input_text):
    """Build the full prompt for a given task.

    For text_processing tasks: wraps input_text in the prompt template.
    For generation tasks: returns input_text directly (it IS the prompt).
    """
    if task not in TASKS:
        raise ValueError(f'Unknown task: {task}. Available: {list(TASKS.keys())}')

    info = TASKS[task]

    if info.get('type') == 'generation':
        return input_text

    return info['prompt_template'].format(input_text=input_text)
