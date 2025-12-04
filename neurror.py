
import time
import json
import os
from sentence_transformers import SentenceTransformer 
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier #To support liblinear, from crashing after developer's update


#These imports exist to let me actually use the tools that they offer
#Before i can use their objects or functions, i have to specify that they must be brought into here

#Before running the code, run this in the terminal: pip install -r dependencies.txt
model = SentenceTransformer("all-MiniLM-L6-v2") 
# I will use variable model, to store which type of pre-trained model I will apply in my code
# Make sure to install Python´s SentenceTransformer, import it, and store it before you apply it in your code



storage = "nrrordata.json" # Here I'm storing the file nrrordata.json
                           # as an object that my python code can interact with
                           # nrrordata.json points you to stored user_inputs

data = {"user_input": []}  # In order for our json file to write user_inputs into its dictionary list
                           # We have to declare it a variable, so our code can interact with it as an object
                           

# These example sentences will be used to train Neurror chat on 4 emotion categories
# Each emotion category has sentence examples
# The model i'm using: sentence transformer already has semantic "closeness" scale pre-trained into it.
# What it doesn't have is: emotion categories, and how much each sentence belongs in said category
# Which is why you must insert your own example data, that the transformer model will be trained upon
# It will do that by noticing patterns in the examples. The model uses 384 dimensions per vector to 
# Describe to itself how close it is to a certain word, category or sentence. 
# Meaning: Every word, sentence or category will have 384 numbers, to describe the object to the model
# Those dimensions may not do much on their own, but when grouped together the bigger picture allows us to
# see how close in meaning and semantics, our sentences are. 
# Your sentences:
# "I am happy", "I'm sad", "I'm joyful"
# The positive sentences will have their numeric dimensions be closer: such as 1 and 2
# But the negative sentences, will be 6. 1 and 2 are closer, than 1 and 6. 
# This is how the model understands sentences, by turning them into integers.
# 384 embeddings total, do not define, but do indicate our model's token limitations.
# In this case: 256 tokens roughly = 1,024 characters. 
# 1 token = 4 characters
# It is not to say that it cannot work with large datasets or inputs
# However, to avoid the program crashing, we must ensure the model processes it chunk by chunk. 
# Once it reaches 256 tokens, there will be a seperate embedding for that next chunk of text
# Afterwards, all of the embeddings are compared and averaged out into a seperate embedding, using pooling (math)
# This approach has negative effects on accuracy, because to split the chunks means to lose context across embeddings
# and land at a general understanding of the text. Which is why it's best to stay
# within the scope of 256 tokens per input, to enhance answer accuracy. 
joy_sentences = [
    "I am so happy with life, I feel free and light",
    "I am so joyful, ecstatic and excited about everything",
    "I love my life and the people around me",
    "I like swinging on a chair every evening",
    "I'm smiling and laughing",
    "He is the most loving person I ever met",
    "She is so optimistic and strong, and enthusiastic"
    "I feel so happy today.",
    "Everything seems bright and warm.",
    "I’m smiling for no reason.",
    "I feel light and free.",
    "Life feels wonderful right now.",
    "I’m filled with excitement.",
    "I feel joy bubbling inside me.",
    "I’m grateful for everything in my life.",
    "Today feels magical.",
    "I’m laughing nonstop.",
    "Everything feels possible.",
    "I’m full of positive energy.",
    "I feel loved and connected.",
    "I’m so proud of myself.",
    "I feel refreshed and hopeful.",
    "I’m glowing with happiness.",
    "My heart feels full.",
    "I feel bright and optimistic.",
    "I’m enjoying every moment today.",
    "I feel like dancing.",
    "I’m thrilled about the future.",
    "I feel peaceful and content.",
    "I’m surrounded by good vibes.",
    "I’m excited for what’s next.",
    "I feel joyful and alive.",
    "I’m bursting with enthusiasm.",
    "Everything feels like a gift.",
    "My spirit feels light.",
    "I’m riding a wave of joy.",
    "I’m so energized today.",
    "Happiness is overflowing in me.",
    "I feel incredibly lucky.",
    "I’m feeling proud and confident.",
    "Today feels joyful and bright.",
    "I’m inspired and excited.",
    "I feel like celebrating.",
    "I’m enjoying myself so much.",
    "I feel cheerful and vibrant.",
    "I’m thankful for this moment.",
    "I’m so happy I could cry.",
    "Life feels beautiful right now.",
    "I feel completely at ease.",
    "Everything around me feels warm.",
    "I’m full of laughter today.",
    "I feel joy spreading in my chest.",
    "I’m delighted by everything.",
    "I feel hopeful and light-hearted.",
    "My day is filled with sunshine.",
    "I’m overflowing with positivity.",
    "I feel uplifted and inspired."
]

sadness_sentences = [
    "I feel horrible, and empty",
    "My life feels like it fell apart",
    "I cannot imagine what life has to offer, everything feels pointless",
    "I am terrible at life, I cannot decide anything, I’m just so defeated and upset",
    "Ugh I feel sick and I’m in so much pain. Somebody end this",
    "I'm crying and cannot get out of bed",
    "I cannot move, it feels too difficult. I’m so tired"
    
    "I feel so empty inside.",
    "Everything feels heavy.",
    "I can’t stop feeling down.",
    "I’m exhausted emotionally.",
    "I feel broken in ways I can’t explain.",
    "My chest hurts from sadness.",
    "I feel like crying all day.",
    "I’m overwhelmed with sorrow.",
    "I don’t feel like myself anymore.",
    "Everything feels pointless.",
    "I’m stuck in a dark place.",
    "I feel hopeless right now.",
    "I’m losing motivation to do anything.",
    "I feel drained and defeated.",
    "My heart feels shattered.",
    "I feel sad even when nothing is wrong.",
    "I’m struggling to get out of bed.",
    "I feel like giving up.",
    "I feel weighed down by everything.",
    "I’m sad and I don’t know why.",
    "I feel numb and distant.",
    "I’m hurting more than I show.",
    "I feel like no one understands me.",
    "Everything feels like too much.",
    "I’m disappointed in myself.",
    "I feel lost and confused.",
    "My sadness feels endless.",
    "I feel overwhelmed by my emotions.",
    "I feel like I’m fading away.",
    "I’m trying but everything feels hard.",
    "I feel stuck in my mind.",
    "I feel like crying all the time.",
    "I’m aching inside.",
    "I’m sad and tired.",
    "I feel like nothing will get better.",
    "My heart feels heavy with grief.",
    "I’m struggling to function.",
    "I feel tearful and lonely.",
    "Everything makes me emotional.",
    "I’m drowning in sadness.",
    "I feel distant from the world.",
    "I’m hurting quietly.",
    "I feel hopeless and worn out.",
    "I’m overwhelmed by pain.",
    "I feel fragile and broken.",
    "Nothing is bringing me joy.",
    "I feel emotionally exhausted.",
    "I feel defeated and small.",
    "I’m sad and I can’t hide it.",
    "I feel like laying down and disappearing."


]

anger_sentences = [
    "UGH! AGH! I’m so furious, I could destroy everything around me.",
    "If I find you I will kill you, I hate you",
    "I’m so tired, and angry at everyone, I’m so irritated and upset",
    "I wanna fight, or run away. I hate it here, fuck all of you",
    "My skin is boiling, and my face is red",
    "I shall revenge upon all of you, I resent you."
    "I’m so irritated I could scream.",
    "Everything is making me furious today.",
    "I can’t stand this situation anymore.",
    "I feel like punching a wall.",
    "I’m so fed up with everyone.",
    "My patience is completely gone.",
    "I’m boiling inside right now.",
    "I feel rage in my chest.",
    "I’m shaking because I’m so angry.",
    "This makes me absolutely livid.",
    "I’m sick of being treated like this.",
    "I hate everything about this moment.",
    "I feel like exploding.",
    "I could tear this room apart.",
    "I’m furious and no one gets it.",
    "My anger is out of control.",
    "I feel betrayed and enraged.",
    "I want to scream until my lungs burn.",
    "I can’t control the fire in me.",
    "I’m irritated beyond belief.",
    "Stop pushing me — I’m losing it.",
    "I’m so mad I can barely think.",
    "Nothing is going the way I want.",
    "I’m clenching my fists in anger.",
    "I feel trapped and furious.",
    "This is pushing me over the edge.",
    "Why does everything make me angry?",
    "I’m so done with all of this.",
    "I feel hostility pouring out of me.",
    "I want to yell at someone.",
    "My blood is boiling.",
    "I feel so disrespected and mad.",
    "I’m annoyed at every little thing.",
    "My jaw hurts from holding in anger.",
    "I want to smash something.",
    "I feel rage rising in my throat.",
    "I’m furious with how things turned out.",
    "This is absolutely infuriating.",
    "I can’t get this anger out of me.",
    "I feel like shouting at the world.",
    "I’m burning up with frustration.",
    "I’m outraged and tired of pretending otherwise.",
    "I feel attacked and angry.",
    "I’m pissed off for no reason.",
    "I’m raging inside even if I look calm.",
    "Everything feels like a personal offense.",
    "I’ve had enough — I’m exploding inside.",
    "I’m sick of holding back my anger.",
    "I feel anger pulsing through me.",
    "I’m furious and I don’t know how to stop it."

]

fear_sentences = [
    "What if nothing will work out, I’m so anxious. What if I will fail and lose everything",
    "Do people around me care? I’m so scared, that everyone will leave me and I will die alone",
    "Oh no! What if I die alone. I cannot do this, I have to freeze, stay in one place and not move",
    "I need to hide or run away, this cannot happen! This is worst case scenario.",
    "My heart is racing, my thoughts are scattered, I feel broken and all over the place."
        
    "I’m terrified something bad will happen.",
    "My heart won’t stop racing.",
    "I feel like danger is everywhere.",
    "I’m scared of losing everything.",
    "I feel frozen with fear.",
    "I can’t stop imagining the worst.",
    "I’m afraid people will leave me.",
    "I feel like I’m being watched.",
    "Something feels very wrong.",
    "I’m shaking because I’m so scared.",
    "I’m afraid to make the wrong choice.",
    "I’m scared I’m not safe.",
    "I feel overwhelmed with anxiety.",
    "I’m afraid to speak or move.",
    "I feel trapped by my own thoughts.",
    "I’m terrified I’ll mess everything up.",
    "I feel like running away.",
    "I’m scared of being alone.",
    "I feel like hiding from the world.",
    "I’m afraid everything will collapse.",
    "I can’t handle this fear in my chest.",
    "I feel like something terrible is coming.",
    "I’m nervous and uneasy.",
    "I’m afraid that I’m not enough.",
    "My stomach sinks with fear.",
    "I’m worried I’ll fail badly.",
    "I feel consumed by panic.",
    "I’m afraid something is wrong with me.",
    "I can’t calm down — I’m too scared.",
    "I feel dread creeping up on me.",
    "I’m afraid of confrontation.",
    "I’m anxious about every little thing.",
    "I keep imagining horrible outcomes.",
    "I feel unsafe and vulnerable.",
    "I’m worried everyone is judging me.",
    "I fear being abandoned.",
    "I’m scared to face tomorrow.",
    "I feel like something bad is hiding around the corner.",
    "I’m afraid of disappointing everyone.",
    "My mind is spiraling with fear.",
    "I feel threatened even when nothing is happening.",
    "I’m scared of losing control.",
    "I’m worried I’ll never calm down.",
    "I feel like crying from fear.",
    "I’m scared of making decisions.",
    "I feel constantly on edge.",
    "I’m worried I’ll get hurt.",
    "I feel like I’m in danger even when I’m not.",
    "This fear won’t let go of me.",
    "I’m terrified and I don’t know why."



]
#################################################################
# Use this to correctly locate our storage file JSON:

#print("json path:", os.path.abspath(storage)) 
####################################################################





 ###### PRE-NLP TEXT CLEAN-UP ######

# Before we process user input and store it in the JSON File
def clean_text(text): #Define our function as clean_text, that accepts one paremeter: variable text
            
    text = text.lower().strip() #### The Information in the text variable will be turned to lower letters, and blank spaces on both ends will be stripped
    text = text.split() # Every word in this variable will be split into seperate strings
    text = " ".join(text) # Every string inside the variable will be joined into one string, using the " " as a glue
            
    return text # In the end we return the text variable to whoever called this function
    
    ######################################  

    
    ##### NEURROR MODEL FUNCTION #####    
def nrr_chat(clean_input):  # Here we define our function as nrr_chat that accepts a variable as paremeter
                            # The variable clean_input contains the refined text of user's input

        vector = model.encode(clean_input) # Here we declare that the variable Vector
                                           # Must take processed input
                                           # And turn the strings into an encoding
                                           # Using the rules of our SentenceTransformer model (384-dimensional vector)
        return vector # Now return the vector to whoever called the nrr_chat function
    
    ##################################### 

    #This is where the neurror engine exists
def neurror():
    print(
        "\n"
        "------------------\n"
        "\n"
        "Welcome to Neurror.\n"
        "\n"
        "Tell me about the emotions you went through today.\n"
        "Or type 'x' to exit."
         "\n"
         "\n"
         "------------------\n"
    )
    
   ################## Training Data #####################
   # This is the point in which we begin to train our data 
   # Before we can actually apply neurror chat as users

    emotion_label = [] # Here we will store our vectors (x,y) from the training data
    emotion_content = [] # Every sentence will be embedded into 384 dimensional vectors (x)
                         # And every sentence will have its emotion label (y), where it belongs

    for sentence in joy_sentences:           #A loop that takes every item
                                             # from the training data variables
                                             # Stores clean sentences in variable "refined"
                                             # Per Sentence is turned into an embedding
                                             # Per item Sentence is added to the emotional_content coordinate
                                             # Per item (sentence) has a label appended to it
        refined = clean_text(sentence)
        emb = model.encode(refined)
        emotion_content.append(emb)
        emotion_label.append("JOY")

    for sentence in sadness_sentences:
        refined = clean_text(sentence)
        emb = model.encode(refined)
        emotion_content.append(emb)
        emotion_label.append("SAD")

    for sentence in anger_sentences:
        refined = clean_text(sentence)
        emb = model.encode(refined)
        emotion_content.append(emb)
        emotion_label.append("ANGER")

    for sentence in fear_sentences:
        refined = clean_text(sentence)
        emb = model.encode(refined)
        emotion_content.append(emb)
        emotion_label.append("FEAR")
    ###########################################

    ##################################################
    # Observe in the terminal whether all 4 emotion labels
    # Have 384-dimensional vectors listed in them
    # If the number is not 384, something is off:

    #for i in range(4):        # i = 0, 1, 2, 3
        #print(f"Embedding: vector dimension of item {i} :", len(emotion_content[i]))
    #print("Total embeddings in all items together:", len(emotion_content))
    #####################################################
    

    # This is the part where we run the training engine
    clf = OneVsRestClassifier((LogisticRegression(max_iter=1000, solver="liblinear")))# Here we have our training "classifier", object
                                                                # 
                                                               # that is meant to use Logistic Regression method, specifically.
                                                               # As a way of training upon our data, and learning
                                                               # How to classify our emotion content, into each emotion label
                                                               # Here we have set our model to do a max of 1000 iterations before convergence
                                                               # This means that the model can make a max of 1000 predictions, and adjusting of its weights
                                                               # For the whole database 
                                                               # Until we reach convergence: Which is when the model looks at the bigger picture
                                                               # Of our dataset and says: Is this the closest we can get?
                                                               # I will also use liblinear, as it is most compatible with small databases
                                                               # Liblinear uses all examples in the dataset to update its weights during iteration

    clf.fit(emotion_content, emotion_label) # Fit emotion_content/label database into the training model 

    # -----------------------------
    # LOOP
    # -----------------------------
    while True:            
        user_input = input("Type here: ")  # As long as user doesn't type x the loop will persist
        print("")
        if user_input.lower() == "x":
            print("You have exited the Neurror program.")
            return  # <-- exits ENTIRE FUNCTION

        clean_input = clean_text(user_input) # Refine user's input before sending to the model

        try: #this is the code we are going to try and execute
            vector = nrr_chat(clean_input) # Run nrr_chat function using user's input, and store it in the vector (It will be stored as list of decimal integers between 0-1)
            probs = clf.predict_proba([vector])[0] # Predict the probability that this sample, in the row 0, belongs to each one of the possible classes (labels)
            classes = clf.classes_ # Use attribute .classes to store all the data labels, inside the classes variable

            print("----------------------\n",
                "Emotion percentages:",
                "\n"
                "----------------------\n")
            for label, pct in zip(classes, probs * 100): #For every label, and its percentage
                print(f"{label}: {round(pct)}%")  #Turn it into a string 
                print("")

        except Exception as fail: # Except if an error arises
            print("Neurror couldn't understand you, Try again")
            print(fail)
            continue # Resume the try-section

        data["user_input"].append(clean_input) #Store cleaned user input in nrrdata.json file

        with open(storage, "w") as f: # We open the file from storage variable, and hold it as an f for writing in data
            json.dump(data, f, indent=4) # Write the data variable into object f (the real file), use indentation: 4

        time.sleep(5) #Allow 5 seconds before next user input
        print(
             "\n",
            "----------------------\n",
            "Talk to Neurror some more, or type 'x' to exit the program",
              "\n")
    
# START PROGRAM
neurror() # Call our function so the user always starts in neurror environment, whenever running the program

    