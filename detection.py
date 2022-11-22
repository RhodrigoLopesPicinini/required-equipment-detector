import tensorflow as tf                                                 # lib do tensorflow para a utilização da maioria das opções
import os                                                               # lib padrão
import cv2                                                              # lib cv2 para detecção de imagem
import numpy as np                                                      # lib numpy para converter dados em arrays e integers
from badgeDetector import badgeDetector                                 # função para detecção de cores e qrcode
from playsound import playsound                                         # lib para tocar arquivos de som
from datetime import datetime                                           # lib para detectar a data do OS
from object_detection.utils import config_util                          # Utilitie para configuração do modelo de detecção
from object_detection.utils import label_map_util                       # Utilitie para o mapa de rotulação do modelo
from object_detection.utils import visualization_utils as viz_utils     # Utilitie para overlay do objeto detectado
from object_detection.builders import model_builder                     # Utilitie para buildar os arquivos referents ao modelo

# Endereços de diretórios importantes para o carregamento do modelo de detecção
# Pipeline.config que no qual possui os comandos e variáveis de orientação para o comportamento do modelo de detecção
CONFIG_PATH = 'Tensorflow/workspace/models/my_ssd_mobnet/pipeline.config'
CHECKPOINT_PATH = 'Tensorflow/workspace/models/my_ssd_mobnet'
ANNOTATION_PATH = 'Tensorflow/workspace/annotations'

# Import da configuração necessária para a utilização do modelo de detecção
config = config_util.get_configs_from_pipeline_file(CONFIG_PATH)

# Carrega a pipeline do modelo de detecção e criação de variável com o modelo pronto
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Checkpoint do treino do modelo
# Checkpoints contém parâmetros originados do treinamento do modelo de detecção
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-11')).expect_partial()

# Função para detecção (uso do modelo)
@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)                   # pré-processamento de imagem (320x320 resize)
    prediction_dict = detection_model.predict(image, shapes)            # previsões do modelo sobre os objetos detectados
    detections = detection_model.postprocess(prediction_dict, shapes)   # pós-processamento das previsões do modelo e imagem
    return detections                                                   # retorna as detecções de objetos pelo modelo

# Índice dos objetos rotulados ({1: {'id': 1, 'name': 'without_mask_and_cap'}, 2: {'id': 2, 'name': 'with_mask_and_cap'}})
category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH+'/label_map.pbtxt')

# Captura de video utilizando OpenCV e alterando a largura e altura do video
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

count = 0
tries = 150
approved = []
# Utilização do Modelo de Detecção de Máscara e Touca Descartável
while True: 
    
    # Frame por frame do vídeo capturado e convertendo para um numpy array
    ret, frame = cap.read()
    image_np = np.array(frame)
    

    # Transforma o numpy array em um tensor e insere o array obtido na função de detecção do modelo
    # Tensor comporta números flutuantes e inteiros que representam formas e dimensionalidades, utilizado para input e output de dados em machine learning
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    
    # Adquirindo o número de objetos detectados, passando para um array e obtendo os detection classes (ids dos objetos detectados)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1

    # copia do array de cada frame para transformação e criação do overlay de detecções
    image_np_with_detections = image_np.copy()

    # Visualização das bounding boxes + scores da detecção (porcentagem)
    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,                           # Passando a cópia da variável de frames 
                detections['detection_boxes'],                      # Desenho das caixas de detecção
                detections['detection_classes']+label_id_offset,    # Identificação do id de cada objeto categorizado para detecção
                detections['detection_scores'],                     # Porcentagem da detecção do objeto
                category_index,                                     # índice das categorias de objeto
                use_normalized_coordinates=True,                    # Coordenadas da detecção para posicionamento do  overlay
                max_boxes_to_draw=1,                                # Máximo de overlays do vídeo
                min_score_thresh=.55,                               # Mínimo da porcentagem de detecção para o overlay aparecer
                agnostic_mode=False)                                # Não será utilizado mask

    # Multiplica os scores para melhor visualização de porcentagem    
    my_scores = detections['detection_scores']*100
    # Ids dos modelos
    my_classes = detections['detection_classes']+label_id_offset
    min_score_thresh=55.0
    
    # Identifica a classe que está sendo detectada através dos scores
    class_detected = ([{'class':category_index.get(value)} 
        for index,value in enumerate(my_classes) 
        if my_scores[index] > min_score_thresh
    ])

    # Se a lista não conter a string "temporary pass", detectar faces
    # Se a lista conter a string "temporary pass", detectar cracha
    if not approved: 
        # Começa a contagem para manter a detecção dependendo da face com equipamento ou sem
        # Quando a contagem chegar a 70, a face que estiver sem equipamento terá seu acesso negado
        # Quando a contagem chegar a 70, a face que estiver com o equipamento receberá o passe temporário para a próxima etapa
        if class_detected:
            tries = 100
            if class_detected[0]['class']['name'] == 'without_mask_and_cap':
                count += 1
                if count == 80:
                    playsound('sounds/failed.wav')
                    count = 0
            elif class_detected[0]['class']['name'] == 'with_mask_and_cap':
                count += 1
                if count == 80:
                    playsound('sounds/success.wav')
                    approved.append('temporary pass')
                    print('Equipamento validado... etapa de confirmação de crachá...')

                    count = 0
        else:
            count = 0
    elif approved:
        # etapa de detecção do cracha, verficando pela cor e QRcode correto, dentro de um determinado limite de tempo
        # Se o cracha for validado, o acesso será garantido e o programa irá imprimir o registro da pessoa e o programa irá retornar para a primeira etapa
        # Caso o cracha for invaludado, o acesso não será garantido e o programa irá retornar para a primeira etapa
        is_badge_verified = badgeDetector(frame)
        if is_badge_verified:
            playsound('sounds/success.wav')

            current_date_and_time = datetime.now()
            print(f'Acesso Permitido: Cientista Marcos, data: {current_date_and_time}')

            approved.pop()
        elif is_badge_verified == False:
            playsound('sounds/failed.wav')

            approved.pop()
        else:
            tries -= 1
            if tries == 0:
                playsound('sounds/failed.wav')

                approved.pop()

    # Janela para visualização do detector de objetos em ação
    image_np_with_detections_resized = cv2.resize(image_np_with_detections, (800, 600))
    cv2.imshow('Disposable Mask and Cap Detector', image_np_with_detections_resized)
    
    # Fechar programa
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        break