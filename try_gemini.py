from dotenv import load_dotenv
import os

load_dotenv()

import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])
response = chat.send_message(
    """You are a helpful AI bot that answers questions for a user. Keep your response short and direct. Here is a transcript of a patient's visit to the doctor:

[doctor] hi , andrew , how are you ?
[patient] hi . good to see you .
[doctor] it's good to see you as well . so i know that the nurse told you about dax . i'd like to tell dax a little bit about you .
[patient] sure .
[doctor] okay ? so , andrew is a 62-year-old male with a past medical history significant for a kidney transplant , hypothyroidism , and arthritis , who presents today with complaints of joint pain . andrew , what's going on with your joint ? what happened ?
[patient] uh , so , over the the weekend , we've been moving boxes up and down our basements stairs , and by the end of the day my knees were just killing me .
[doctor] okay . is , is one knee worse than the other ?
[patient] equally painful .
[doctor] okay .
[patient] both of them .
[doctor] and did you , did you injure one of them ?
[patient] um , uh , i've had some knee problems in the past but i think it was just the repetition and the weight of the boxes .
[doctor] okay . all right . and , and what have you taken for the pain ?
[patient] a little tylenol . i iced them for a bit . nothing really seemed to help , though .
[doctor] okay . all right . um , and does it prevent you from doing , like , your activities of daily living , like walking and exercising and things like that ?
[patient] uh , saturday night it actually kept me up for a bit . they were pretty sore .
[doctor] mm-hmm . okay . and any other symptoms like fever or chills ?
[patient] no .
[doctor] joint pain ... i mean , like muscle aches ?
[patient] no .
[doctor] nausea , vomiting , diarrhea ?
[patient] no .
[doctor] anything like that ?
[patient] no .
[doctor] okay . all right . now , i know that you've had the kidney transplant a few years ago for some polycystic kidneys .
[patient] mm-hmm .
[doctor] um , how are you doing with that ? i know that you told dr. gutierrez-
[patient] mm .
[doctor] . a couple of weeks ago .
[patient] yes .
[doctor] everything's okay ?
[patient] so far , so good .
[doctor] all right . and you're taking your immunosuppressive medications ?
[patient] yes , i am .
[doctor] okay . all right . um , and did they have anything to say ? i have n't gotten any reports from them , so ...
[patient] no , n- nothing out of the ordinary , from what they reported .
[doctor] okay . all right . um , and in terms of your hyperthyroidism , how are you doing with the synthroid ? are you doing okay ?
[patient] uh , yes , i am .
[doctor] you're taking it regularly ?
[patient] on the clock , yes .
[doctor] yes . okay . and any fatigue ? weight gain ? anything like that that you've noticed ?
[patient] no , nothing out of the ordinary .
[doctor] okay . and just in general , you know , i know that we've kind of battled with your arthritis .
[patient] mm-hmm .
[doctor] you know , it's hard because you ca n't take certain medications 'cause of your kidney transplant .
[patient] sure .
[doctor] so other than your knees , any other joint pain or anything like that ?
[patient] every once in a while , my elbow , but nothing , nothing out of the ordinary .
[doctor] okay . all right . now i know the nurse did a review of systems sheet when you checked in . any other symptoms i might have missed ?
[patient] no .
[doctor] no headaches ?
[patient] no headaches .
[doctor] anything like that w- ... okay . all right . well , i wan na go ahead and do a quick physical exam , all right ? hey , dragon , show me the vital signs . so here in the office , your vital signs look good . you do n't have a fever , which is good .
[patient] mm-hmm .
[doctor] your heart rate and your , uh , blood pressure look fine . i'm just gon na check some things out , and i'll let you know what i find , okay ?
[patient] perfect .
[doctor] all right . does that hurt ?
[patient] a little bit . that's tender .
[doctor] okay , so on physical examination , on your heart exam , i do appreciate a little two out of six systolic ejection murmur-
[patient] mm-hmm .
[doctor] . which we've heard in the past . okay , so that seems stable . on your knee exam , there is some edema and some erythema of your right knee , but your left knee looks fine , okay ? um , you do have some pain to palpation of the right knee and some decreased range of motion , um , on exam , okay ? so what does that mean ? so we'll go ahead and we'll see if we can take a look at some of these things . i know that they did an x-ray before you came in , okay ?
[patient] mm-hmm .
[doctor] so let's take a look at that .
[patient] sure .
[doctor] hey , dragon , show me the right knee x-ray . so here's the r- here's your right knee x-ray . this basically shows that there's good bony alignment . there's no acute fracture , which is not surprising , based on the history .
[patient] mm-hmm .
[doctor] okay ? hey , dragon , show me the labs . and here , looking at your lab results , you know , your white blood cell count is not elevated , which is good . you know , we get concerned about that in somebody who's immunocompromised .
[patient] mm-hmm .
[doctor] and it looks like your kidney function is also very good . so i'm , i'm very happy about that .
[patient] yeah .
[doctor] okay ? so i just wan na go over a little bit about my assessment and my plan for you .
[patient] mm-hmm .
[doctor] so for your knee pain , i think that this is an acute exacerbation of your arthritis , okay ? so i wan na go ahead and if ... and prescribe some ultram 50 milligrams every six hours as needed .
[patient] okay .
[doctor] okay ? i also wan na go ahead and just order an autoimmune panel , okay ? hey , dragon , order an autoimmune panel . and you know , i , i want , i want you to just take it easy for right now , and if your symptoms continue , we'll talk about further imaging and possibly referral to physical therapy , okay ?
[patient] you got it .
[doctor] for your second problem , your hypothyroidism , i wan na go ahead and continue you on this ... on the synthroid , and i wan na go ahead and order some thyroid labs , okay ?
[patient] sure .
[doctor] hey , dragon , order a thyroid panel . and then for your last problem , the arthritis , you know , we just kinda talked about that . you know , it's gon na be a struggle for you because again , you ca n't take some of those anti-inflammatory medications because of your kidney transplant , so ...
[patient] mm-hmm .
[doctor] you know , let's see how we do over the next couple weeks , and again , we'll refer you to physical therapy if we need to , okay ?
[patient] you got it .
[doctor] you have any questions ?
[patient] not at this point .
[doctor] okay . hey , dragon , finalize the note .

What is the first name of the patient?

Don't give information outside the document or repeat your findings."""
)
print(response.text)
