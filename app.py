import streamlit as st
import streamlit.components.v1 as components 
import cv2 
import numpy as np 
from ultralytics import YOLO
import streamlit_option_menu as option_menu
from PIL import Image, ImageDraw
import io
import tempfile
import imageio.v2 as imageio
from moviepy.editor import ImageSequenceClip
import os
import shutil
# from ultralytics.yolo.utils.plotting import Annotator
from cv2 import cvtColor
import os

#Importing the model

model = YOLO('best.pt')
def bgr2rgb(image):
    return image[:, :, ::-1]


    
def process_video(video_path):
    # Load the video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30  # Set a default value for fps if it is 0 or None

    # Create a list to store the processed frames
    processed_frames = []

    # Process each frame in the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform the prediction on the frame
        prediction = model.predict(frame)
        frame_with_bbox = prediction[0].plot()

        # Convert the frame to PIL Image and store in the list
        processed_frames.append(Image.fromarray(frame_with_bbox))

    cap.release()

    # Create the output video file path
    video_path_output = "output.mp4"

    # Save the processed frames as individual images
    with tempfile.TemporaryDirectory() as temp_dir:
        for i, frame in enumerate(processed_frames):
            frame.save(f"{temp_dir}/frame_{i}.png")

        # Create a video clip from the processed frames
        video_clip_path = f"{temp_dir}/clip.mp4"
        os.system(f"ffmpeg -framerate {fps} -i {temp_dir}/frame_%d.png -c:v libx264 -pix_fmt yuv420p {video_clip_path}")

        # Rename the video clip with the desired output path
        shutil.copy2(video_clip_path, video_path_output)

    return video_path_output



        
def main():

    with open("styles.css", "r") as source_style:
        st.markdown(f"<style>{source_style.read()}</style>", 
             unsafe_allow_html = True)
        
    st.title("Pet Skin Disease Classification")
    Header = st.container()
    js_code = """
        const elements = window.parent.document.getElementsByTagName('footer');
        elements[0].innerHTML = "Pet Skin Disease Classification " + new Date().getFullYear();
        """
    st.markdown(f"<script>{js_code}</script>", unsafe_allow_html=True)
            
    #st.image("logo.png")
    
    ##MainMenu
    
    with st.sidebar:
        selected = option_menu.option_menu(
            "Main Menu",
            options=[
                "Project Information",
                "Classify Pet Skin Disease",
                "Contributors"
            ],
        )
    
    st.sidebar.markdown('---')
        
    ##HOME page 
    
    if selected == "Project Information":
        
        st.subheader("Problem Statement")
        problem_statement = """
        Our project focuses on automating pet skin disease classification using deep learning techniques.
        By leveraging convolutional neural networks and a diverse dataset, we aim to develop an accurate and efficient model. 
        This innovation holds the promise of improving pet healthcare and diagnosis.
        """
        
        st.write(problem_statement)
        
        with st.expander("Read More"): 
            text = """
        Our project is dedicated to automating pet skin disease classification through the utilization of cutting-edge
        deep learning techniques, specifically Convolutional Neural Networks (CNNs), and a comprehensive dataset. 
        This initiative has the potential to bring about a profound transformation in the field of pet healthcare.
        By accurately and efficiently classifying pet skin diseases, we aim to improve diagnosis, treatment, and overall pet welfare."""
            
            st.write(text)
        
        st.subheader("Our Solution")
        Project_goal = """
        Our Team developed a Machine Learning ( ML ) model based on the YOLOv8 Architecture, which was trained on a comprehensive
        dataset of pet skin disease images and manually annotated them to highlight the various types of classification. Once the model was trained,
        we proceeded to test its performance on new and unseen data. This testing phase was vital to ensure that our model could
        generalize well and accurately identify skin diseases in real-world scenarios. In addition to the model,
        we developed a web application using the Streamlit API which serves as a user-friendly interface for others to test the 
        trained model on their own pet images and videos
        """
        st.write(Project_goal)
        
    elif selected == "Classify Pet Skin Disease": 
        
        text1="Make the settings in the left Panel and see the classification"
        st.write(text1)
        st.sidebar.subheader('Settings')
        
        options = st.sidebar.radio(
            'Options:', ('Image', 'Video'), index=1)
        
        st.sidebar.markdown("---")
         # Image
        if options == 'Image':
            upload_img_file = st.sidebar.file_uploader('Upload Image', type=['jpg', 'jpeg', 'png'])
            if upload_img_file is not None:
                file_bytes = np.asarray(bytearray(upload_img_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, 1)
                img_rgb = cvtColor(img, cv2.COLOR_BGR2RGB)
                prediction = model.predict(img_rgb)
                res_plotted = prediction[0].plot()
                image_pil = Image.fromarray(res_plotted)
                image_bytes = io.BytesIO()
                image_pil.save(image_bytes, format='PNG')

                # Create a container for the two images side by side
                col1, col2 = st.columns(2)

                # Display the uploaded image in the first column
                col1.image(img_rgb, caption='Uploaded Image', use_column_width=True)

                # Display the predicted image in the second column
                col2.image(image_bytes, caption='Predicted Image', use_column_width=True)
                
                predicted_classes = prediction[0].names[0] if len(prediction) > 0 else []
        
        
                st.subheader(f"Predicted Disease: {predicted_classes}")
                disease_info = {
                'Dog leprosy': {
                    'Diagnosis': 'Identifying dog leprosy typically involves a biopsy and culture of the affected tissue to confirm the presence of the bacteria.',
                    'Treatment': 'Treatment usually involves long-term antibiotic therapy, including drugs like clarithromycin, rifampicin, and clofazimine.',
                    'Management': 'Besides antibiotics, supportive care such as wound management and pain relief may be necessary to alleviate symptoms.',
                    'Monitoring': 'Regular veterinary check-ups are essential to monitor the progress of treatment and adjust medications as needed.',
                    'Prevention': 'Preventive measures, including maintaining good hygiene and avoiding contact with infected animals, may help reduce the risk of transmission.'
                },
                'Cat-Leprosy': {
                    'Diagnosis': 'Diagnosis typically involves a biopsy and culture of affected tissue to identify the bacteria.',
                    'Treatment': 'Treatment often consists of a prolonged course of antibiotics, such as clarithromycin, doxycycline, and rifampicin, prescribed by a veterinarian.',
                    'Duration': 'The duration of treatment varies depending on the severity of the infection and the cat\'s response to antibiotics.',
                    'Supportive Care': 'Supportive care, including wound management and pain relief, may be necessary alongside antibiotic therapy.',
                    'Monitoring': 'Regular follow-up visits with a veterinarian are crucial to monitor the cat\'s progress and adjust the treatment plan as needed.'
                },
                'Dog-ringworm': {
                    'Symptoms': 'Signs include circular patches of hair loss, redness, and scaling on the skin. Dogs may also scratch or lick the affected areas.',
                    'Diagnosis': 'Diagnosis often involves a fungal culture or microscopic examination of skin samples to confirm the presence of the fungus.',
                    'Treatment': 'Treatment typically involves antifungal medications, such as topical creams, shampoos, or oral medications like griseofulvin or itraconazole.',
                    'Environmental Control': 'It\'s essential to disinfect the dog\'s living environment to prevent the spread of the fungus to other pets or humans.',
                    'Prevention': 'Preventive measures include maintaining good hygiene, avoiding contact with infected animals, and promptly treating any skin issues to prevent ringworm spread. Regular grooming and cleaning of bedding can also help prevent re-infection.'
                },
                'Ear-mites': {
                    'Symptoms': 'Signs of ear mite infestation include itching, head shaking, ear scratching, redness, and dark discharge in the ears.',
                    'Diagnosis': 'Diagnosis is usually based on clinical signs and the presence of mites seen under a microscope during an ear examination.',
                    'Treatment': 'Treatment typically involves cleaning the ears to remove debris and applying prescription ear drops or medications to kill the mites. Common medications may contain ingredients like ivermectin, selamectin, or milbemycin oxime.',
                    'Environmental Control': 'It\'s essential to clean the pet\'s bedding and living areas to prevent re-infestation.',
                    'Prevention': 'Preventive measures include regular ear cleaning, routine veterinary check-ups, and prompt treatment of any signs of ear discomfort or irritation. Regular grooming can also help prevent ear mite infestations.'
                },
                'cat-ringworm': {
                    'Symptoms': 'Signs include circular patches of hair loss, redness, and scaling on the skin. Cats may also scratch or lick the affected areas.',
                    'Diagnosis': 'Diagnosis often involves a fungal culture or microscopic examination of skin samples to confirm the presence of the fungus.',
                    'Treatment': 'Treatment typically involves antifungal medications, such as topical creams, shampoos, or oral medications like griseofulvin or itraconazole. Additionally, clipping long-haired cats may aid in treatment.',
                    'Environmental Control': 'It\'s crucial to disinfect the cat\'s living environment to prevent the spread of the fungus to other pets or humans.',
                    'Prevention': 'Preventive measures include maintaining good hygiene, avoiding contact with infected animals, and promptly treating any skin issues to prevent the spread of ringworm. Regular grooming and cleaning of bedding can also help prevent re-infection.'
                }
                }
                        
                for key, value in disease_info[predicted_classes].items():
                    st.write(f"**{key}:** {value}")



                
        if options == 'Video':
            upload_vid_file = st.sidebar.file_uploader(
                'Upload Video', type=['mp4', 'avi', 'mkv']
                )
            if upload_vid_file is not None:
            # Save the uploaded video file temporarily
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                temp_file.write(upload_vid_file.read())

                # Process the video frames and get the output video file path
                video_path_output = process_video(temp_file.name)

                # Display the processed video using the st.video function
                st.video(video_path_output)
                
                

                # Remove the temporary files
                temp_file.close()
                os.remove(video_path_output)
           
                 
            
    elif selected == "Contributors":
        st.subheader("Contributors")
        st.markdown("<b><u>Project Contributors :</u></b> \n  ", unsafe_allow_html=True)
        st.write("""1.  Panistha Gupta \n 2.  Varun Venkat Sarvanan \n   """)

if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
