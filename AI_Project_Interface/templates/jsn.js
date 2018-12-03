
window.SpeechRecognition = window.webkitSpeechRecognition || window.SpeechRecognition;
const synth = window.speechSynthesis;
const recognition = new SpeechRecognition();
recognition.lang = "en-US";
var answer = "";



//Test for output txt file
// var txt = '';
// var xmlhttp = new XMLHttpRequest();
// xmlhttp.onreadystatechange = function(){
//   if(xmlhttp.status == 200 && xmlhttp.readyState == 4){
//     txt = xmlhttp.responseText;
//   }
// };
// xmlhttp.open("GET","output_trial.txt",true);
// xmlhttp.send();


//     // stop link reloading the page
//  event.preventDefault();
// }





//test for button 
function output_to_box() {
			   document.getElementById('output_text').value = document.getElementById('input_text').value;
			   //Debugging purpose
          console.log(document.getElementById('output_text').value);
          
 }

function startDictation() {

    if (window.hasOwnProperty('webkitSpeechRecognition')) {
      //Start recognition process
      recognition.start();
      

      recognition.onresult = function(event) {
      	//Set result to a constant
      	const speechToText = event.results[0][0].transcript;
      	answer = speechToText;
      	check_parsing()
      	//Set input text value to the input text
        document.getElementById('input_text').value = speechToText;

        recognition.stop();
       										 }; //function onresult

      recognition.onerror = function(event) {
        recognition.stop();
      										} //function if recognition failture

    }
  } //end dictation function





