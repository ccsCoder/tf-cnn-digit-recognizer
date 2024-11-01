const quotes = {
  "00:00": [
    {
      time: "midnight",
      book: "The Picture of Dorian Gray",
      author: "Oscar Wilde",
      prefix: "As ",
      suffix:
        " was striking bronze blows upon the dusky air, Dorian Gray, dressed commonly, and with a muffler wrapped round his throat, crept quietly out of his house.",
    },
    {
      time: "midnight",
      book: "Ulysses",
      author: "James Joyce",
      prefix: '"But wait till I tell you," he said. :We had a ',
      suffix:
        " lunch too after all the jollification and when we sallied forth it was blue o'clock the morning after the night before\"",
    },
    {
      time: "Midnight",
      book: "Pale Fire",
      author: "Vladimir Nabokov",
      prefix: '"',
      suffix:
        '," you said. What is midnight to the young? And suddenly a festive blaze was flung Across five cedar trunks, snow patches showed, And a patrol car on our bumpy road Came to a crunching stop. Retake, retake!',
    },
    {
      time: "12.00 pm",
      book: "A Squatter's Tale",
      author: "Ike Oguine",
      prefix:
        "That a man who could hardly see anything more than two feet away from him could be employed as a security guard suggested to me that our job was not to secure anything but to report for work every night, fill the bulky ledger with cryptic remarks like 'Patrolled perimeter ",
      suffix:
        ", No Incident' and go to the office every fortnight for our wages and listen to the talkative Ms Elgassier.",
    },
    {
      time: "midnight",
      book: "A Nocturnal upon St Lucy's Day",
      author: "John Donne",
      prefix: "'Tis the year's ",
      suffix:
        ", and it is the day's, Lucy's, who scarce seven hours herself unmasks; The sun is spent, and now his flasks Send forth light squibs, no constant rays;",
    },
    {
      time: "midnight",
      book: "Beauty and Sadness",
      author: "Yasunari Kawabata",
      prefix: "At ",
      suffix:
        " his wife and daughter might still be bustling about, preparing holiday delicacies in the kitchen, straightening up the house, or perhaps getting their kimonos ready or arranging flowers. Oki would sit in the living room and listen to the radio. As the bells rang he would look back at the departing year. He always found it a moving experience.",
    },
    {
      time: "twelve",
      book: "Hamlet",
      author: "Shakespeare",
      prefix: "Bernardo: 'Tis now struck ",
      suffix: "; get thee to bed, Francisco.",
    },
    {
      time: "midnight",
      book: "Nights At The Circus",
      author: "Angela Carter",
      prefix:
        "Big Ben concluded the run-up, struck and went on striking. (...) But, odder still - Big Ben had once again struck ",
      suffix:
        ". The time outside still corresponded to that registered by the stopped gilt clock, inside. Inside and outside matched exactly, but both were badly wrong. H'm.",
    },
    {
      time: "midnight",
      book: "Molloy",
      author: "Samuel Beckett",
      prefix:
        "But in the end I understood this language. I understood it, I understood it, all wrong perhaps. That is not what matters. It told me to write the report. Does this mean I am freer now than I was? I do not know. I shall learn. Then I went back into the house and wrote, It is ",
      suffix:
        ". The rain is beating on the windows. It was not midnight. It was not raining.",
    },
    {
      time: "0000h.",
      book: "Infinite Jest",
      author: "David Foster Wallace",
      prefix: "Cartridges not allowed after ",
      suffix: ", to encourage sleep.",
    },
    {
      time: "twelve",
      book: "Hamlet",
      author: "William Shakespeare",
      prefix:
        "Francisco. You come most carefully upon your hour. Bernardo. 'Tis now struck ",
      suffix: ". Get thee to bed, Francisco.",
    },
    {
      time: "0000h",
      book: "Infinite Jest",
      author: "David Foster Wallace",
      prefix:
        "Gately can hear the horns and raised voices and u-turn squeals way down below on Wash. That indicate it's around ",
      suffix: "., the switching hour.",
    },
    {
      time: "twelve",
      book: "Hamlet",
      author: "William Shakespeare",
      prefix: "Hamlet: What hour now? Horatio: I think it lacks of ",
      suffix: ". Marcellus: No, it is struck.",
    },
    {
      time: "midnight",
      book: "Swann's Way",
      author: "Marcel Proust",
      prefix:
        "He is certain he heard footsteps: they come nearer, and then die away. The ray of light beneath his door is extinguished. It is ",
      suffix:
        "; some one has turned out the gas; the last servant has gone to bed, and he must lie all night in agony with no one to bring him any help.",
    },
    {
      time: "midnight",
      book: "Behind the Scenes at the Museum",
      author: "Kate Atkinson",
      prefix: "I am conceived to the chimes of ",
      suffix:
        " on the clock on the mantelpiece in the room across the hall. The clock once belonged to my great-grandmother (a woman called Alice) and its tired chime counts me into the world.",
    },
    {
      time: "twelve",
      book: "Wuthering Heights",
      author: "Emily Brontë",
      prefix:
        "I took her hand in mine, and bid her be composed; for a succession of shudders convulsed her frame, and she would keep straining her gaze towards the glass. 'There's nobody here!' I insisted. 'It was YOURSELF, Mrs. Linton: you knew it a while since.' 'Myself!' she gasped, 'and the clock is striking ",
      suffix: "! It's true, then! that's dreadful!'",
    },
    {
      time: "midnight",
      book: "Midnight's Children",
      author: "Salman Rushdie",
      prefix: "I was born in the city of Bombay ... On the stroke of ",
      suffix:
        ", as a matter of fact. Clock-hands joined palms in respectful greeting as I came. Oh, spell it out, spell it out: at the precise instant of India's arrival at independence, I tumbled forth into the world.",
    },
    {
      time: "midnight",
      book: "Molloy",
      author: "Samuel Beckett",
      prefix: "It is ",
      suffix:
        ". The rain is beating on the windows. I am calm. All is sleeping. Nevertheless I get up and go to my desk. I can't sleep. ...",
    },
    {
      time: "midnight",
      book: "Harry Potter and the Half-Blood Prince",
      author: "JK Rowling",
      prefix: "It was nearing ",
      suffix:
        " and the Prime Minister was sitting alone in his office, reading a long memo that was slipping through his brain without leaving the slightest trace of meaning behind.",
    },
    {
      time: "Midnight",
      book: "Oliver Twist",
      author: "Charles Dickens",
      prefix: "",
      suffix:
        " had come upon the crowded city. The palace, the night-cellar, the jail, the madhouse; the chambers of birth and death, of health and sickness; the rigid face of the corpse and the calm sleep of the child - midnight was upon them all",
    },
    {
      time: "Midnight",
      book: "After Dark",
      author: "Murakami",
      prefix: "",
      suffix:
        " is approaching, and while the peak of activity has passed, the basal metabolism that maintains life continues undiminished, producing the basso continuo of the city's moan, a monotonous sound that neither rises or falls but is pregnant with foreboding",
    },
    {
      time: "midnight",
      book: "The Raven",
      author: "Edgar Allan Poe",
      prefix: "Once upon a ",
      suffix:
        " dreary, while I pondered weak and weary, Over many a quaint and curious volume of forgotten lore, While I nodded, nearly napping, suddenly there came a tapping, As of some one gently rapping, rapping at my chamber door. `'Tis some visitor,' I muttered, `tapping at my chamber door - Only this, and nothing more.'",
    },
    {
      time: "twelve",
      book: "Dr Faustus",
      author: "Christopher Marlowe",
      prefix: "The clock striketh ",
      suffix:
        " O it strikes, it strikes! Now body, turn to air, Or Lucifer will bear thee quick to hell. O soul, be changed into little water drops, And fall into the ocean, ne'er to be found. My God, my God, look not so fierce on me!",
    },
    {
      time: "midnight",
      book: "The Life and Opinions of Tristram Shandy, Gentleman",
      author: "Laurence Sterne",
      prefix:
        "The first night, as soon as the corporal had conducted my uncle Toby up stairs, which was about 10 - Mrs. Wadman threw herself into her arm chair, and crossing her left knee with her right, which formed a resting-place for her elbow, she reclin'd her cheek upon the palm of her hand, and leaning forwards, ruminated until ",
      suffix: " upon both sides of the question.'",
    },
    {
      time: "twelve o'clock",
      book: "David Copperfield",
      author: "Charles Dickens",
      prefix:
        "To begin my life with the beginning of my life, I record that I was born (as I have been informed an believe) on a Friday, at ",
      suffix:
        " at night. It was remarked that the clock began to strike, and I began to cry, simultaneously.",
    },
    {
      time: "midnight",
      book: "Henry IV",
      author: "William Shakespeare",
      prefix: "We have heard the chimes at ",
      suffix: ".",
    },
  ],
  "00:01": [
    {
      time: "one minute past midnight",
      book: "Death at Midnight",
      author: "Donald A. Cabana",
      prefix: "With the appointed execution time of ",
      suffix:
        " just seconds away, I knocked on the metal door twice. The lock turned and the door swiftly swung open.",
    },
  ],
  "00:02": [
    {
      time: "Two minutes past midnight",
      book: "Night of the Krait",
      author: "Shashi Warrier",
      prefix: "",
      suffix:
        ". With me in the lead the fourteen other men of Teams Yellow, White and Red moved out of the clearing and separated for points along the wall where they would cross over into the grounds",
    },
  ],
  "00:03": [
    {
      time: "after twelve o'clock",
      book: "The Ragged Trousered Philanthropists",
      author: "Robert Tressell",
      prefix: "It was ",
      suffix:
        " when Easton came home. Ruth recognised his footsteps before he reached the house, and her heart seemed to stop beating when she heard the clang of the gate, as it closed after he had passed through.",
    },
    {
      time: "three minutes past midnight",
      book: "Since Ibsen",
      author: "George Jean Nathan",
      prefix: "It was just ",
      suffix:
        " when I last saw Archer Harrison alive. I remember, because I said it was two minutes past and he looked at his watch and said it was three minutes past.",
    },
    {
      time: "Three minutes after midnight.",
      book: "The Historian",
      author: "Elizabeth Kostova",
      prefix:
        "Suddenly I felt a great stillness in the air, then a snapping of tension. I glanced at my watch. ",
      suffix:
        " I was breathing normally and my pen moved freely across the page. Whatever stalked me wasn’t quite as clever as I’d feared, I thought, careful not to pause in my work.",
    },
  ],
  "00:04": [
    {
      time: "four minutes past midnight",
      book: "Anzio: Epic of Bravery",
      author: "Fred Sheehan",
      prefix: "At ",
      suffix:
        ", January 22, Admiral Lowry's armada of more than 250 ships reached the transport area off Anzio. The sea was calm, the night was black.",
    },
  ],
  "00:05": [
    {
      time: "0005h",
      book: "Infinite Jest",
      author: "David Foster Wallace",
      prefix:
        "E.M. Security, normally so scrupulous with their fucking trucks at ",
      suffix:
        "., is nowhere around, lending weight to yet another cliché. If you asked Gately what he was feeling right this second he'd have no idea.",
    },
  ],
  "00:06": [
    {
      time: "six minutes past midnight",
      book: "West of Hell's Fringe",
      author: "Glenn Shirley",
      prefix: "At ",
      suffix: ", death relieved the sufferer.",
    },
  ],
  "00:07": [
    {
      time: "seven minutes after midnight",
      book: "The Curious Incident of the Dog in the Night-Time",
      author: "Mark Haddon",
      prefix: "It was ",
      suffix:
        ". The dog was lying on the grass in the middle of the lawn in front of Mrs Shears' house. Its eyes were closed. It looked as if it was running on its side, the way dogs run when they think they are chasing a cat in a dream.",
    },
  ],
  "00:08": [
    {
      time: "eight past midnight",
      book: "The Brigantine",
      author: "James Pascoe",
      prefix:
        '"Hour of the night!" exclaimed the priest; "it is day, not night, and the hour is ',
      suffix: '!"',
    },
  ],
  "00:09": [
    {
      time: "12.09am",
      book: "The Bhutto Murder Trail: From Waziristan to GHQ",
      author: "Amir Mir",
      prefix: "At ",
      suffix:
        " on 18 October, the cavalcade had reached the Karsaz Bridge, still ten kilometres from her destination.",
    },
  ],
  "00:10": [
    {
      time: "ten minutes past midnight",
      book: "The Queue",
      author: "Jonathan Barrow",
      prefix: "It was at ",
      suffix:
        ". Three police cars, Alsations and a black maria arrive at the farmhouse. The farmer clad only in a jock-strap, refused them entry.",
    },
  ],
  "00:11": [
    {
      time: "eleven minutes past midnight",
      book: "The Longest Night",
      author: "Gavin Mortimer",
      prefix:
        "The first incendiaries to hit St Thomas's Hospital had splattered Riddell House at ",
      suffix:
        ", from where a few hours earlier the Archbishop of Canterbury had given 'an inspiring address'.",
    },
  ],
  "00:12": [
    {
      time: "0 Hours, 12 Minutes",
      book: "Mason & Dixon",
      author: "Thomas Pynchon",
      prefix: "Clock time is ",
      suffix:
        ", 0 Seconds. Twenty three minutes later, they have their first sight of Venus. Each lies with his Eye clapp'd to the Snout of an identical two and a half foot Gregorian reflector made by Mr Short, with Darkening-Nozzles by Mr Bird.",
    },
    {
      time: "twelve minutes past midnight",
      book: "Kentucky heat",
      author: "Fern Michaels",
      prefix: "It was ",
      suffix:
        " when mother and daughter saw the first lightning strike. It hit the main barn with such force the ground trembled under their feet.",
    },
  ],
  "00:14": [
    {
      time: "fourteen minutes past midnight",
      book: "The Matarese Circle",
      author: "Robert Ludlum",
      prefix: "It was exactly ",
      suffix:
        " when he completed the final call. Among the men he had reched were honourable men. Their voices would be heard by the President.",
    },
  ],
  "00:15": [
    {
      time: "twelve-fifteen",
      book: "Watchers",
      author: "Dean Koontz",
      prefix: "At ",
      suffix:
        " he got out of the van. He tucked the pistol under the waistband of his trousers and crossed the silent, deserted street to the Hudston house.",
    },
    {
      time: "twelve-fifteen",
      book: "Watchers",
      author: "Dean Koontz",
      prefix: "At ",
      suffix:
        " he got out of the van. He tucked the pistol under the waistband of his trousers and crossed the silent, deserted street to the Hudston house. He let himself through an unlocked wooden gate onto a side patio brightened only by moonlight filtered through the leafy branches of an enormous sheltering coral tree. He paused to pull on a pair of supple leather gloves.",
    },
  ],
  "00:16": [
    {
      time: "sixteen minutes past midnight",
      book: "The Longest Night",
      author: "Gavin Mortimer",
      prefix: "At ",
      suffix: ", Block 4 was hit and the roof set alight.",
    },
  ],
  "00:17": [
    {
      time: "seventeen minutes after twelve",
      book: "Vanvild Kava",
      author: "Isaac Bashevis Singer",
      prefix:
        "Kava ordered two glasses of coffee for himself and his beloved and some cake. When the pair left, exactly ",
      suffix: ", the club began to buzz with excitement.",
    },
  ],
  "00:18": [
    {
      time: "12.18am",
      book: "The Book of Lies",
      author: "Mary Horlock",
      prefix: "21st December 1985, ",
      suffix:
        " [In bed] Michael doesn’t believe in Heaven or Hell. He’s got closer to death than most living people and he tells me there was no tunnel of light or dancing angels. I’m a bit disappointed, to be honest.",
    },
  ],
  "00:20": [
    {
      time: "twelve-twenty",
      book: "The Quiet American",
      author: "Graham Greene",
      prefix:
        "Now she was kneading the little ball of hot paste on the convex margin of the bowl and I could smell the opium. There is no smell like it. Beside the bed the alarm-clock showed ",
      suffix: ", but already my tension was over. Pyle had diminished.",
    },
  ],
  "00:21": [
    {
      time: "12.21am",
      book: "The Raw Shark Texts",
      author: "Steven Hall",
      prefix:
        "Nobody had been one of Mycroft Ward's most important operatives and for sixty seconds every day, between ",
      suffix:
        " and 12.22am., his laptop was permitted to connect directly with the gigantic online database of self that was Mycroft Ward's mind.",
    },
  ],
  "00:22": [
    {
      time: "12.22am.",
      book: "The Raw Shark Texts",
      author: "Steven Hall",
      prefix:
        "Nobody had been one of Mycroft Ward's most important operatives and for sixty seconds every day, between 12.21am and ",
      suffix:
        ", his laptop was permitted to connect directly with the gigantic online database of self that was Mycroft Ward's mind.",
    },
  ],
  "00:23": [
    {
      time: "twenty-three minutes past midnight",
      book: "The Tin Drum",
      author: "Günter Grass",
      prefix:
        "Oskar weighed the wristwatch in his hand, then gave the rather fine piece with its luminous dial showing ",
      suffix:
        " to little Pinchcoal. He looked up inquiringly at his chief. Störtebeker nodded his assent. And Oskar said, as he adjusted his drum snugly for the trip home: 'Jesus will lead the way. Follow thou me!'",
    },
  ],
  "00:24": [
    {
      time: "12.24am",
      book: "The Longest Night",
      author: "Gavin Mortimer",
      prefix: "Sanders with Sutton as his gunner began their patrol at ",
      suffix: ", turning south towards Beachy Head at 10,000 ft.",
    },
  ],
  "00:25": [
    {
      time: "five-and-twenty minutes past midnight",
      book: "Fruitfulness",
      author: "Emile Zola",
      prefix:
        "Charlotte remembered that she had heard Gregoire go downstairs again, almost immediately after entering his bedroom, and before the servants had even bolted the house-doors for the night. He had certainly rushed off to join Therese in some coppice, whence they must have hurried away to Vieux-Bourg station which the last train to Paris quitted at ",
      suffix: ". And it was indeed this which had taken place.",
    },
    {
      time: "Twenty-five past midnight",
      book: "The Soldier's Wife",
      author: "Joanna Trollope",
      prefix: "I mean, look at the time! ",
      suffix: "! It was a triumph, it really was!",
    },
  ],
  "00:26": [
    {
      time: "12.26am.",
      book: "Bryant & May Off the Rails",
      author: "Christopher Fowler",
      prefix:
        "\"A Mr Dutta from King's Cross called and told me you were on your way. He said you wanted to see the arrival of yesterday's ",
      suffix:
        " It'll take me a few minutes to cue up the footage. Our regular security bloke isn't here today; he's up before Haringey Magistrates' Court for gross indecency outside the headquarters of the Dagenham Girl Pipers.\"",
    },
  ],
  "00:28": [
    {
      time: "12.28",
      book: "11/22/63",
      author: "Stephen King",
      prefix: "The DRINK CHEER-UP COFFEE wall clock read ",
      suffix: ".",
    },
  ],
  "00:29": [
    {
      time: "Twenty-nine minutes past twelve",
      book: "The Mark of the Knife",
      author: "Clayton H Ernst",
      prefix:
        '"What time is it?" asked Teeny-bits. The station agent hauled out his big silver watch, looked at it critically and announced: "',
      suffix: '.” “Past twelve!" repeated Teeny-bits. "It can\'t be."',
    },
  ],
  "00:30": [
    {
      time: "half-past twelve",
      book: "The Amateur Cracksman",
      author: "E.W. Hornung",
      prefix: "It was ",
      suffix:
        " when I returned to the Albany as a last desperate resort. The scene of my disaster was much as I had left it. The baccarat-counters still strewed the table, with the empty glasses and the loaded ash-trays. A window had been opened to let the smoke out, and was letting in the fog instead.",
    },
  ],
  "00:31": [
    {
      time: "00:31",
      book: "Zombie Apocalypse! Fightback",
      author: "Stephen Jones",
      prefix:
        "Third individual approaches unnoticed and without caution. Once within reach, individual reaches out toward subjects. Recording terminates: timecode: ",
      suffix: ":02.",
    },
  ],
  "00:32": [
    {
      time: "Thirty-two minutes past midnight",
      book: "Ixtapa",
      author: "Everette Howard Hunt",
      prefix: "",
      suffix:
        "; the way things were going I could be at it all night. Before beginning a completely new search of the dial I had a thought: maybe this safe didn't open on zero as older models did, but on a factory-set number",
    },
  ],
  "00:33": [
    {
      time: "thirty-three minutes past midnight",
      book: "Cover her Face",
      author: "P.D. James",
      prefix:
        '"So that at twelve-thirty-three you bolted the south door?" "Yes," replied Stephen Maxie easily. "At ',
      suffix: '."',
    },
  ],
  "00:34": [
    {
      time: "Thirty-four minutes past midnight",
      book: "Killer Tune",
      author: "Dreda Say Mitchell",
      prefix: "",
      suffix:
        ". 'We got ten minutes to be back here.' LT didn't argue. Schoolboy knew his former trade. LT's eyes fretted over the museum. 'Not still worrying about the security, are you, because there ain't none.",
    },
  ],
  "00:40": [
    {
      time: "twenty to one",
      book: "A Subaltern's Love Song",
      author: "John Betjeman",
      prefix: "We sat in the car park till ",
      suffix: "/ And now I'm engaged to Miss Joan Hunter Dunn.",
    },
  ],
  "00:42": [
    {
      time: "eighteen minutes to one",
      book: "Marjorie Morningstar",
      author: "Herman Wouk",
      prefix:
        "The butt had been growing warm in her fingers; now the glowing end stung her skin. She crushed the cigarette out and stood, brushing ash from her black skirt. It was ",
      suffix:
        ". She went to the house phone and called his room. The telephone rang and rang, but there was no answer.",
    },
  ],
  "00:43": [
    {
      time: "Twelve-forty-three",
      book: "A Pocket Full of Rye",
      author: "Agatha Christie",
      prefix:
        "Died five minutes ago, you say? he asked. His eye went to the watch on his wrist. ",
      suffix: ", he wrote on the blotter.",
    },
  ],
  "00:45": [
    {
      time: "12.45",
      book: "Pig and Pepper: A Comedy of Youth",
      author: "David Footman",
      prefix: "At ",
      suffix:
        ", during a lull, Mr Yoshogi told me that owing to the war there were now many more women in England than men.",
    },
    {
      time: "third quarter after midnight",
      book: "The Reef",
      author: "Edith Wharton",
      prefix:
        "At the thought he jumped to his feet and took down from its hook the coat in which he had left Miss Viner's letter. The clock marked the ",
      suffix:
        ", and he knew it would make no difference if he went down to the post-box now or early the next morning; but he wanted to clear his conscience, and having found the letter he went to the door.",
    },
  ],
  "00:47": [
    {
      time: "12:47a.m",
      book: "Last Night I Dreamed Of Peace",
      author: "Andrew X. Pham",
      prefix: "At ",
      suffix: ", Uncle Ho left us forever.",
    },
  ],
  "00:50": [
    {
      time: "12.50",
      book: "Three Men in a Boat",
      author: "Jerome K Jerome",
      prefix: "The packing was done at ",
      suffix:
        "; and Harris sat on the big hamper, and said he hoped nothing would be found broken. George said that if anything was broken it was broken, which reflection seemed to comfort him. He also said he was ready for bed.",
    },
  ],
  "00:54": [
    {
      time: "six minutes to one",
      book: "A Double Barrelled Detective Story",
      author: "Mark Twain",
      prefix:
        "Everybody was happy; everybody was complimentary; the ice was soon broken; songs, anecdotes, and more drinks followed, and the pregnant minutes flew. At ",
      suffix:
        ", when the jollity was at its highest— BOOM! There was silence instantly.",
    },
  ],
  "00:55": [
    {
      time: "Five to one",
      book: "61 Hours",
      author: "Lee Child",
      prefix:
        "He rolled one way, rolled the other, listened to the loud tick of the clock, and was asleep a minute later. ",
      suffix: " in the morning. Fifty-one hours to go.",
    },
  ],
  "00:56": [
    {
      time: "12:56 A.M.",
      book: "Extremely Loud and Incredibly Close",
      author: "Jonathan Safran Foer",
      prefix: "It was ",
      suffix:
        " when Gerald drove up onto the grass and pulled the limousine right next to the cemetery.",
    },
    {
      time: "12:56",
      book: "Lessons in Essence",
      author: "Dana Standridge",
      prefix:
        "Teacher used to lie awake at night facing that clock, batting his eyelashes against his pillowcase to mimic the sound of the rolling drop action. One night, and this first night is lost in the countless later nights of compounding wonder, he discovered a game. Say the time was ",
      suffix: ".",
    },
  ],
  "00:57": [
    {
      time: "12:57",
      book: "Lessons in Essence",
      author: "Dana Standridge",
      prefix: "A minute had passed, and the roller dropped a new leaf. ",
      suffix:
        ". 12 + 57 = 69; 6 + 9 = 15; 1 + 5 = 6. 712 + 5 = 717; 71 + 7 = 78; 7 + 8 = 15; 1 + 5 = 6 again.",
    },
  ],
  "00:58": [
    {
      time: "almost at one in the morning",
      book: "The Idiot",
      author: "Fyodor Dostoyevsky",
      prefix:
        "It was downright shameless on his part to come visiting them, especially at night, ",
      suffix: ", after all that had happened.",
    },
  ],
  "00:59": [
    {
      time: "About one o’clock",
      book: "Freedom",
      author: "Jonathan Frantzen",
      prefix: "‘What time is it now?’ she said. ‘",
      suffix:
        "’. ‘In the morning?’ Herera’s friend leered at her. ‘No, there’s a total eclipse of the sun’.",
    },
  ],
  "01:00": [
    {
      time: "1.00 am.",
      book: "Sister",
      author: "Rosamund Lupton",
      prefix: "",
      suffix: " I felt the surrounding quietness suffocating me",
    },
    {
      time: "one o’clock",
      book: "Atomised",
      author: "Michel Houellebecq",
      prefix:
        "He didn’t know what was at the end of the chute. The opening was narrow (though large enough to take the canary). He dreamed that the chute opened onto vast garbage bins filled with old coffee filters, ravioli in tomato sauce and mangled genitalia. Huge worms, as big as the canary, armed with terrible beaks, would attack the body. Tear off the feet, rip out its intestines, burst the eyeballs. He woke up, trembling; it was only ",
      suffix:
        ". He swallowed three Xanax. So ended his first night of freedom.",
    },
    {
      time: "nearly one o'clock",
      book: "The Woman in White",
      author: "Wilkie Collins",
      prefix:
        "I looked attentively at her, as she put that singular question to me. It was then ",
      suffix:
        ". All I could discern distinctly by the moonlight was a colourless, youthful face, meagre and sharp to look at about the cheeks and chin; large, grave, wistfully attentive eyes; nervous, uncertain lips; and light hair of a pale, brownish-yellow hue.",
    },
    {
      time: "one in the morning",
      book: "Tomorrow",
      author: "Graham Swift",
      prefix:
        "I'm the only one awake in this house on this night before the day that will change all our lives. Though it's already that day: the little luminous hands on my alarm clock (which I haven't set) show just gone ",
      suffix: ".",
    },
    {
      time: "One am",
      book: "London Belongs to Me",
      author: "Norman Collins",
      prefix: "It was the thirtieth of May by now. ",
      suffix:
        " on the thirtieth of May 1940. Quite a famous date on which to be lying awake and staring at the ceiling. Already in the creeks and tidal estuaries of England the pleasure-boats and paddle-steamers were casting their moorings for the day trip to Dunkirk. And, over on the other side, Ted stood as a good a chance as anyone else.",
    },
    {
      time: "one",
      book: "Hamlet",
      author: "William Shakespeare",
      prefix:
        "Last night of all, When yon same star that's westward from the pole Had made his course t'illume that part of heaven Where now it burns, Marcellus and myself, The bell then beating ",
      suffix: " -",
    },
    {
      time: "one o'clock in the morning",
      book: "The Long Dark Tea-time of the Soul",
      author: "Douglas Adams",
      prefix:
        "The station was more crowded than he had expected to find it at - what was it? he looked up at the clock - ",
      suffix:
        ". What in the name of God was he doing on King's Cross station at one o'clock in the morning, with no cigarette and no home that he could reasonably expect to get into without being hacked to death by a homicidal bird?",
    },
  ],
  "01:01": [
    {
      time: "About one o’clock",
      book: "Freedom",
      author: "Jonathan Frantzen",
      prefix: "‘What time is it now?’ she said. ‘",
      suffix:
        "’. ‘In the morning?’ Herera’s friend leered at her. ‘No, there’s a total eclipse of the sun’.",
    },
  ],
  "01:06": [
    {
      time: "1:06",
      book: "No Country for Old Men",
      author: "Cormac McCarthy",
      prefix: "When he woke it was ",
      suffix:
        " by the digital clock on the bedside table. He lay there looking at the ceiling, the raw glare of the vaporlamp outside bathing the bedroom in a cold and bluish light. Like a winter moon.",
    },
  ],
  "01:08": [
    {
      time: "1.08a.m.",
      book: "The Rosie Project",
      author: "Graeme Simsion",
      prefix: "It was ",
      suffix:
        " but he had left the ball at the same time as I did, and had further to travel.",
    },
  ],
  "01:09": [
    {
      time: "nine minutes past one",
      book: "The Black Bag",
      author: "Louis Joseph Vance",
      prefix:
        "They made an unostentatious exit from their coach, finding themselves, when the express had rolled on into the west, upon a station platform in a foreign city at ",
      suffix: " o'clock in the morning - but at length without their shadow.",
    },
  ],
  "01:10": [
    {
      time: "1.10am",
      book: "South: The Endurance Expedition",
      author: "Ernest Shackleton",
      prefix: "February 26, Saturday - Richards went out ",
      suffix:
        " and found it clearing a bit, so we got under way as soon as possible, which was 2:10a.m.",
    },
  ],
  "01:11": [
    {
      time: "nearer to one than half past",
      book: "The Affair at the Victory Ball",
      author: "Agatha Christie",
      prefix:
        "Declares one of the waiters was the worse for liquor, and that he was giving him a dressing down. Also that it was ",
      suffix: ".",
    },
  ],
  "01:12": [
    {
      time: "1:12am",
      book: "The Curious Incident of the Dog in the Night-Time",
      author: "Mark Haddon",
      prefix: "It was ",
      suffix:
        " when Father arrived at the police station. I did not see him until 1:28am but I knew he was there because I could hear him. He was shouting, 'I want to see my son,' and 'Why the hell is he locked up?' and, 'Of course I'm bloody angry.'",
    },
  ],
  "01:15": [
    {
      time: "quarter past one",
      book: 'My Life and Hard Times: "The Night the Ghost Got In"',
      author: "James Thurber",
      prefix:
        "I am sorry, therefore, as I have said, that I ever paid any attention to the footsteps. They began about a ",
      suffix:
        " o'clock in the morning, a rhythmic, quick-cadenced walking around the dining-room table.",
    },
    {
      time: "1.15am.",
      book: "Sour Sweet",
      author: "Timothy Mo",
      prefix:
        "Lily Chen always prepared an 'evening' snack for her husband to consume on his return at ",
      suffix: "",
    },
    {
      time: "quarter past one",
      book: 'My Life and Hard Times: "The Night the Ghost Got In"',
      author: "James Thurber",
      prefix:
        "The ghost that got into our house on the night of November 17, 1915, raised such a hullabaloo of misunderstandings that I am sorry I didn't just let it keep on walking, and go to bed. Its advent caused my mother to throw a shoe through a window of the house next door and ended up with my grandfather shooting a patrolman. I am sorry, therefore, as I have said, that I ever paid any attention to the footsteps. They began about a ",
      suffix:
        " o'clock in the morning, a rhythmic, quick-cadenced walking around the dining-room table.",
    },
  ],
  "01:16": [
    {
      time: "sixteen past one",
      book: "Nothing Gold Can Stay",
      author: "Dana Stabenow",
      prefix: "At ",
      suffix: ", they walked into the interview room.",
    },
    {
      time: "1.16am",
      book: "Murder on the Orient Express",
      author: "Agatha Christie",
      prefix: "From 1am to ",
      suffix: " vouched for by other two conductors.",
    },
  ],
  "01:17": [
    {
      time: "seventeen minutes past one",
      book: "A voyage round the moon",
      author: "Jules Verne",
      prefix: "At that moment (it was ",
      suffix:
        " in the morning) Lieutenant Bronsfield was preparing to leave the watch and return to his cabin, when his attention was attracted by a distant hissing noise.",
    },
    {
      time: "1:17",
      book: "The Road",
      author: "Cormac McCarthy",
      prefix: "The clocks stopped at ",
      suffix:
        ". A long shear of light and then a series of low concussions. He got up and went to the window. What is it? she said. He didnt answer. He went into the bathroom and threw the lightswitch but the power was already gone. A dull rose glow in the windowglass. He dropped to one knee and raised the lever to stop the tub and then turned on both taps as far as they would go. She was standing in the doorway in her nightwear, clutching the jamb, cradling her belly in one hand. What is it? she said. What is happening?",
    },
  ],
  "01:20": [
    {
      time: "twenty minutes past one",
      book: "Jeeves and the Feudal Spirit",
      author: "P.G. Wodehouse",
      prefix:
        '"Well!" she said, looking like a minor female prophet about to curse the sins of the people. "May I trespass on your valuable time long enough to ask what in the name of everything bloodsome you think you\'re playing at, young piefaced Bertie? It is now some ',
      suffix:
        " o'clock in the morning, and not a spot of action on your part.\"",
    },
    {
      time: "1.20am",
      book: "The Curious Incident Of The Dog In The Night-Time",
      author: "Mark Haddon",
      prefix: "Then it was ",
      suffix:
        ", but I hadn't heard Father come upstairs to bed. I wondered if he was asleep downstairs or whether he was waiting to come in and kill me. So I got out my Swiss Army Knife and opened the saw blade so that I could defend myself.",
    },
  ],
  "01:22": [
    {
      time: "1:22",
      book: "Extremely Loud and Incredibly Close",
      author: "Jonathan Safran Foer",
      prefix: "It was ",
      suffix: " when we found Dad's grave.",
    },
  ],
  "01:23": [
    {
      time: "twenty-three minutes past one",
      book: "A Mummer's Tale",
      author: "Anatole France",
      prefix: "The clock marked ",
      suffix:
        ". He was suddenly full of agitation, yet hopeful. She had come! Who could tell what she would say? She might offer the most natural explanation of her late arrival.",
    },
  ],
  "01:24": [
    {
      time: "1.24am",
      book: "Body Parts: Essays on Life-Writing",
      author: "Hermione Lee",
      prefix: "Larkin had died at ",
      suffix:
        ", turning to the nurse who was with him, squeezing her hand, and saying faintly, 'I am going to the inevitable.'",
    },
  ],
  "01:25": [
    {
      time: "twenty-five minutes past one",
      book: "The Moonstone",
      author: "Wilkie Collins",
      prefix:
        "He made a last effort; he tried to rise, and sank back. His head fell on the sofa cushions. It was then ",
      suffix: " o'clock.",
    },
  ],
  "01:26": [
    {
      time: "one twenty-six A.M.",
      book: "The Silver Metal Lover",
      author: "Tanith Lee",
      prefix: "When I reached the stop and got off, it was already ",
      suffix: " by the bus's own clock. I had been gone over ten hours.",
    },
  ],
  "01:27": [
    {
      time: "twenty-seven minutes past one",
      book: "Trackers",
      author: "Deon Meyer",
      prefix: "At ",
      suffix: " she felt as if she was levitating out of her body.",
    },
  ],
  "01:28": [
    {
      time: "1:28 am",
      book: "The Curious Incident Of The Dog In The Night-Time",
      author: "Mark Haddon",
      prefix:
        "It was 1:12 am when Father arrived at the police station. I did not see him until ",
      suffix:
        " but I knew he was there because I could hear him. He was shouting, 'I want to see my son,' and 'Why the hell is he locked up?' and, 'Of course I'm bloody angry.'",
    },
  ],
  "01:29": [
    {
      time: "one-twenty-nine A.M.",
      book: "The Narc",
      author: "William Edmund Butterworth",
      prefix: "He exited the men's room at ",
      suffix: "",
    },
  ],
  "01:30": [
    {
      time: "Half-past one",
      book: "Rhapsody on a Windy Night",
      author: "TS Eliot",
      prefix: '"',
      suffix:
        '”, The street lamp sputtered, The street lamp muttered, The street lamp said, "Regard that woman ..."',
    },
    {
      time: "1:30 A.M.",
      book: "Microserfs",
      author: "Douglas Coupland",
      prefix: "Around ",
      suffix:
        " the door opened and I thought it was Karla, but it was Bug, saying Karla and Laura had gone out for a stag night after they ran out of paint.",
    },
    {
      time: "one thirty",
      book: "Gone Tomorrow",
      author: "Lee Child",
      prefix:
        "The late hour helped. It simplified things. It categorized the population. Innocent bystanders were mostly home in bed. I walked for half an hour, but nothing happened. Until ",
      suffix: " in the morning. Until I looped around to 22nd and Broadway.",
    },
    {
      time: "1:30 a.m.",
      book: "Ghostwritten",
      author: "David Mitchell",
      prefix: "The radio alarm clock glowed ",
      suffix:
        " Bad karaoke throbbed through walls. I was wide awake, straightjacketed by my sweaty sheets. A headache dug its thumbs into my temples. My gut pulsed with gamma interference: I lurched to the toilet.",
    },
  ],
  "01:32": [
    {
      time: "One-thirty-two",
      book: "Stamboul Train",
      author: "Graham Greene",
      prefix:
        "She grinned at him with malicious playfulness, showing great square teeth, and then ran for the stairs. ",
      suffix:
        ". She thought that she heard a whistle blown and took the last three steps in one stride.",
    },
  ],
  "01:33": [
    {
      time: "One-thirty-three a.m.",
      book: "Skeletons",
      author: "Kat Fox",
      prefix: "He looked at his watch. ",
      suffix: " He had been asleep on this bench for over an hour and a half.",
    },
  ],
  "01:38": [
    {
      time: "one-thirty-eight",
      book: "The Narc",
      author: "William Edmund Butterworth",
      prefix: "At ",
      suffix:
        " am suspect left the Drive-In and drove to seven hundred and twenty three North Walnut, to the rear of the residence, and parked the car.",
    },
  ],
  "01:40": [
    {
      time: "one-forty am",
      book: "Bones to Ashes",
      author: "Kathy Reichs",
      prefix: "March twelfth, ",
      suffix:
        ", she leaves a group of drinking buddies to catch a bus home. She never makes it.",
    },
  ],
  "01:44": [
    {
      time: "sixteen minutes to two",
      book: "Trackers",
      author: "Deon Meyer",
      prefix:
        "She knew it was the stress, two long days of stress, and she looked at her watch, ",
      suffix:
        ", and she almost leaped with fright, a shock wave rippling through her body, where had the time gone?",
    },
  ],
  "01:46": [
    {
      time: "one forty-six a.m.",
      book: "Fardnock's Revenge",
      author: "J.W. Stockton",
      prefix: "That particular phenomenom got Presto up at ",
      suffix:
        "; silently, he painted his face and naked body with camouflage paint. He opened the door to his room and stepped out into the common lobby.",
    },
  ],
  "01:50": [
    {
      time: "ten minutes before two AM",
      book: "Dog",
      author: "Michelle Herman",
      prefix:
        "No, she thought: every spinster legal secretary, bartender, and orthodontist had a cat or two—and she could not tolerate (not even as a lark, not even for a moment at ",
      suffix: "), embodying cliché.",
    },
  ],
  "01:51": [
    {
      time: "nine minutes to two",
      book: "Trackers",
      author: "Deon Meyer",
      prefix: "At ",
      suffix:
        " the other vehicle arrived. At first Milla didn't believe her eyes: that shape, those markings.",
    },
  ],
  "01:54": [
    {
      time: "Six minutes to two",
      book: "Trackers",
      author: "Dean Koontz",
      prefix: "",
      suffix:
        ". Janina Mentz watched the screen, where the small program window flickered with files scrolling too fast to read, searching for the keyword",
    },
  ],
  "02:00": [
    {
      time: "About two",
      book: "Dead in the water",
      author: "Carola Dunn",
      prefix:
        '"The middle of the night?" Alec asked sharply."Can you be more definite?" "',
      suffix:
        '. Just past." Daisy noted that he expressed no concern for her safety.',
    },
    {
      time: "two o'clock",
      book: "Les Miserables",
      author: "Victor Hugo",
      prefix: "As ",
      suffix: " pealed from the cathedral bell, Jean Valjean awoke.",
    },
    {
      time: "2 A.M.",
      book: "Bech: A Book",
      author: "J. Updike",
      prefix: "Get on plane at ",
      suffix:
        ", amid bundles, chickens, gypsies, sit opposite pair of plump fortune tellers who groan and (very discreetly) throw up all the way to Tbilisi.",
    },
    {
      time: "two",
      book: "Macbeth",
      author: "Shakespeare",
      prefix: "Lady Macbeth: Out, damned spot! out, I say!—One: ",
      suffix:
        ": why, then, 'tis time to do't.—Hell is murky!—Fie, my lord, fie! a soldier, and afeard? What need we fear who knows it, when none can call our power to account?—Yet who would have thought the old man to have had so much blood in him.",
    },
    {
      time: "It struck two.",
      book: "Notes from the underground",
      author: "Fyodor Dostoyevsky",
      prefix:
        "Somewhere behind a screen a clock began wheezing, as though oppressed by something, as though someone were strangling it. After an unnaturally prolonged wheezing there followed a shrill, nasty, and as it were unexpectedly rapid, chime - as though someone were suddenly jumping forward. ",
      suffix:
        " I woke up, though I had indeed not been asleep but lying half-conscious.",
    },
    {
      time: "two o'clock",
      book: "The Brothers Karamazov",
      author: "Fyodor Dostoyevsky",
      prefix:
        "When all had grown quiet and Fyodor Pavlovich went to bed at around ",
      suffix:
        ", Ivan Fyodorovich also went to bed with the firm resolve of falling quickly asleep, as he felt horribly exhausted.'",
    },
  ],
  "02:01": [
    {
      time: "2.01am.",
      book: "The Selected Works of T.S. Spivet",
      author: "Reif Larsen",
      prefix: "I checked my watch. ",
      suffix:
        " The cheeseburger Happy Meal was now only a distant memory. I cursed myself for not also ordering a breakfast sandwich for the morning.",
    },
  ],
  "02:02": [
    {
      time: "almost 2:04",
      book: "Oblivion",
      author: "David Foster Wallace",
      prefix:
        '"Wake up." "Having the worst dream." "I should certainly say you were." "It was awful. It just went on and on." "I shook you and shook you and." "Time is it." "It\'s nearly - ',
      suffix: ".”",
    },
  ],
  "02:04": [
    {
      time: "2:04",
      book: "Oblivion",
      author: "David Foster Wallace",
      prefix:
        '"Wake up." "Having the worst dream." "I should certainly say you were." "It was awful. It just went on and on." "I shook you and shook you and." "Time is it." "It\'s nearly - almost ',
      suffix: ".”",
    },
  ],
  "02:05": [
    {
      time: "2.05",
      book: "London Fields",
      author: "Martin Amis",
      prefix: "At ",
      suffix: " the fizzy tights came crackling off.",
    },
    {
      time: "Five minutes past two",
      book: "The Picture of Dorian Gray",
      author: "Oscar Wilde",
      prefix:
        "Then he began ringing the bell. In about ten minutes his valet appeared, half dressed, and looking very drowsy. ‘I am sorry to have had to wake you up, Francis,’ he said, stepping in; ‘but I had forgotten my latch-key. What time is it?’ ‘",
      suffix:
        ", sir,’ answered the man, looking at the clock and yawning. ‘Five minutes past two? How horribly late! You must wake me at nine to-morrow. I have some work to do.’",
    },
  ],
  "02:07": [
    {
      time: "2:07 a.m.",
      book: "The Curious Incident of the Dog in the Night-Time",
      author: "Mark Haddon",
      prefix: "At ",
      suffix:
        " I decided that I wanted a drink of orange squash before I brushed my teeth and got into bed, so I went downstairs to the kitchen. Father was sitting on the sofa watching snooker on the television and drinking whisky. There were tears coming out of his eyes.",
    },
    {
      time: "2.07 am",
      book: "The Curious Incident of the Dog in the Night-time",
      author: "Mark Haddon",
      prefix: "But I couldn't sleep. And I got out of bed at ",
      suffix:
        " and I felt scared of Mr. Shears so I went downstairs and out of the front door into Chapter Road.",
    },
    {
      time: "2.07 a.m.",
      book: "Before I Go to Sleep",
      author: "S. J. Watson",
      prefix: "Saturday, 17 November — ",
      suffix:
        " I cannot sleep. Ben is upstairs, back in bed, and I am writing this in the kitchen. He thinks I am drinking a cup of cocoa that he has just made for me. He thinks I will come back to bed soon. I will, but first I must write again.",
    },
  ],
  "02:10": [
    {
      time: "Ten minutes past two",
      book: "The Picture of Dorian Gray",
      author: "Oscar Wilde",
      prefix: "“",
      suffix:
        ', sir," answered the man, looking at the clock and blinking. "Ten minutes past two? How horribly late! ..”',
    },
    {
      time: "2:10am",
      book: "South: The Endurance Expedition",
      author: "Ernest Shackleton",
      prefix:
        "Decided to get under way again as soon as there is any clearance. Snowing and blowing, force about fifty or sixty miles an hour. February 26, Saturday - Richards went out 1:10am and found it clearing a bit, so we got under way as soon as possible, which was ",
      suffix: "",
    },
  ],
  "02:12": [
    {
      time: "2.12am",
      book: "The Haunter of the Dark",
      author: "HP Lovecraft",
      prefix: "Then the lights went out all over the city. It happened at ",
      suffix:
        " according to power-house records, but Blake's diary gives no indication of the time. The entry is merely, 'Lights out - God help me.'",
    },
  ],
  "02:13": [
    {
      time: "02.13",
      book: "The Second Internet Cafe, Part 1: The Dimension Researcher",
      author: "Chris James",
      prefix:
        "Now, listen: your destination is Friday, 4 August 1944, and the window will punch through at 22.30 hours. You're going to a dimension that diverged from our own at ",
      suffix:
        " on the morning of Wednesday 20 February 1918, over twenty-six years earlier. You don't know what it's going to be like...",
    },
  ],
  "02:15": [
    {
      time: "2.15am",
      book: "The Shadow Out of Time",
      author: "H.P. Lovecraft",
      prefix: "At ",
      suffix:
        " a policeman observed the place in darkness, but with the stranger's motor still at the curb.",
    },
    {
      time: "two fifteen",
      book: "The Night People",
      author: "Jack Finney",
      prefix: "It did. When the alarm rang at ",
      suffix:
        ", Lew shut it off, snapped on the little bedside lamp, then swung his feet to the floor to sit on the edge of the bed, holding his eyes open.",
    },
  ],
  "02:17": [
    {
      time: "Two-seventeen",
      book: "Freedom",
      author: "Jonathan Franzen",
      prefix:
        '"What time is it now?" He turned her very dusty alarm clock to check. "',
      suffix:
        '," he marveled. It was the strangest time he\'d seen in his entire life. "I apologize that the room is so messy," Lalitha said. "I like it. I love how you are. Are you hungry? I\'m a little hungry." "No, Walter." She smiled. "I\'m not hungry. But I can get you something." "I was thinking, like, a glass of soy milk. Soy beverage."',
    },
    {
      time: "2.17",
      book: "The Ipcress File",
      author: "Len Deighton",
      prefix:
        'One of the "choppers" stopped, did an about-turn and came back to me. The flare spluttered and faded, and now the glare of the spotlight blinded me. I sat very still. It was ',
      suffix:
        ". Against the noise of the blades a deeper resonant sound bit into the chill black air.",
    },
  ],
  "02:18": [
    {
      time: "2:18 in the morning",
      book: "Moo",
      author: "Jane Smiley",
      prefix: "It was ",
      suffix:
        ", and Donna could see no one else in any other office working so late.",
    },
  ],
  "02:20": [
    {
      time: "Two-twenty",
      book: "Southern Mail",
      author: "Antoine de Saint Exupery",
      prefix: "She turned abruptly to the nurse and asked the time. '",
      suffix:
        "' 'Ah...Two-twenty!' Genevieve repeated, as though there was something urgent to be done.",
    },
    {
      time: "two twenty",
      book: "The Night People",
      author: "Jack Finney",
      prefix:
        "The night of his third walk Lew slept in his own apartment. When his eyes opened at ",
      suffix:
        ", by the green hands of his alarm, he knew that this time he'd actually been waiting for it in his sleep.",
    },
  ],
  "02:21": [
    {
      time: "2:21 a.m.",
      book: "The Night of the Generals",
      author: "Hans Hellmut Kirst",
      prefix: "",
      suffix:
        " Lance-Corporal Hartmann emerged from the house in the Rue de Londres",
    },
    {
      time: "Two-twenty-one",
      book: "Hard Boiled Wonderland and the End of the World",
      author: "Haruki Murakami",
      prefix:
        "It was the urge to look up at the sky. But of course there was no sun nor moon nor stars overhead. Darkness hung heavy over me. Each breath I took, each wet footstep, everything wanted to slide like mud to the ground. I lifted my left hand and pressed on the light of my digital wristwatch. ",
      suffix:
        ". It was midnight when we headed underground, so only a little over two hours had passed. We continued walking down, down the narrow trench, mouths clamped tight.",
    },
  ],
  "02:24": [
    {
      time: "2.24am.",
      book: "After You’d Gone",
      author: "Maggie O’Farrell",
      prefix: "It was ",
      suffix:
        " She stumbled out of bed, tripping on her shoes that she’d kicked off earlier and pulled on a jumper.",
    },
  ],
  "02:25": [
    {
      time: "2.25am.",
      book: "Nineteen Eighty-Three: The Red Riding Quartet, Book Four",
      author: "David Peace",
      prefix: "You see it is time: ",
      suffix: " You get out of bed.",
    },
  ],
  "02:26": [
    {
      time: "2.26am",
      book: "The Lighted Rooms",
      author: "Richard Mason",
      prefix: "Listened to a voicemail message left at ",
      suffix: " by Claude.",
    },
  ],
  "02:27": [
    {
      time: "2.27am.",
      book: "One Step Behind",
      author: "Henning Mankell",
      prefix: "The moon didn’t shine again until ",
      suffix:
        " It was enough to show Wallander that he was positioned some distance below the tree.",
    },
  ],
  "02:28": [
    {
      time: "2.28am",
      book: "Mr Commitment",
      author: "Mike Gayle",
      prefix: "",
      suffix: ": Ran out of sheep and began counting other farmyard animals",
    },
  ],
  "02:30": [
    {
      time: "2:30 a.m.",
      book: "The Night People",
      author: "Jack Finney",
      prefix:
        '"Get into the mood, Shirl!" Lew said. "The party\'s already started! Yippee! You dressed for a party, Harry?" "Yep. Something told me to put on dinner clothes when I went to bed tonight." "I\'m in mufti myself: white gloves and matching tennis shoes. But I\'m sorry to report that Jo is still in her Dr. Dentons. What\'re you wearing, Shirl?" "My old drum majorette\'s outfit. The one I wore to the State Finals. Listen, we can\'t tie up the phones like this." "Why not?" said Harry. "Who\'s going to call at ',
      suffix:
        ' with a better idea? Yippee, to quote Lew, we\'re having a party! What\'re we serving, Lew?" "Beer, I guess. Haven\'t got any wine, have we, Jo?" "Just for cooking."',
    },
    {
      time: "half past two",
      book: "The Little Stranger",
      author: "Sarah Waters",
      prefix: "At about ",
      suffix:
        " she had been woken by the creak of footsteps out on the stairs. At first she had been frightened.",
    },
    {
      time: "0230",
      book: "Infinite Jest",
      author: "David Foster Wallace",
      prefix: "Inc, I tried to pull her off about ",
      suffix: ", and there was this fucking… sound.",
    },
    {
      time: "2.30am",
      book: "Any Human Heart",
      author: "William Boyd",
      prefix: "It is ",
      suffix:
        " and I am tight. As a tick, as a lord, as a newt. Must write this down before the sublime memories fade and blur.",
    },
  ],
  "02:31": [
    {
      time: "2.31am.",
      book: "The Curious Incident Of The Dog In The Night-Time",
      author: "Mark Haddon",
      prefix:
        "And then I woke up because there were people shouting in the flat and it was ",
      suffix: " And one of the people was Father and I was frightened.",
    },
  ],
  "02:32": [
    {
      time: "2.32 a.m.",
      book: "The Rosie Project",
      author: "Graeme Simsion",
      prefix: "The last guests departed at ",
      suffix:
        ", two hours and two minutes after the scheduled completion time.",
    },
  ],
  "02:33": [
    {
      time: "Two-thirty-three",
      book: "A Swell-looking Babe",
      author: "Jim Thompson",
      prefix: "But it wasn't going on! It was two-thirty-four, well. ",
      suffix:
        " and nothing had happened. Suppose he got a room call, or the elevator night-bell rang, now.",
    },
  ],
  "02:34": [
    {
      time: "two-thirty-four",
      book: "A Swell-looking Babe",
      author: "Jim Thompson",
      prefix: "But it wasn't going on! It was ",
      suffix:
        ", well. Two-thirty-three and nothing had happened. Suppose he got a room call, or the elevator night-bell rang, now.",
    },
  ],
  "02:35": [
    {
      time: "2.35",
      book: "The Haunter of the Dark",
      author: "HP Lovecraft",
      prefix: "For what happened at ",
      suffix:
        " we have the testimony of the priest, a young, intelligent, and well-educated person; of Patrolman William J. Monohan of the Central Station, an officer of the highest reliability who had paused at that part of his beat to inspect the crowd.",
    },
  ],
  "02:36": [
    {
      time: "2.36am",
      book: "The Ipcress File",
      author: "Len Deighton",
      prefix: "It was about ",
      suffix:
        " when a provost colonel arrived to arrest me. At 2.36 1/2 I remembered the big insulating gauntlets. But even had I remembered before, what could I have done?",
    },
  ],
  "02:37": [
    {
      time: "Thirty-seven minutes past two",
      book: "The Stand",
      author: "Stephen King",
      prefix: "June 13, 1990. ",
      suffix: " in the morning. And sixteen seconds.",
    },
  ],
  "02:43": [
    {
      time: "2:43",
      book: "Neuromancer",
      author: "William Gibson",
      prefix: "She settled back beside him. 'It's ",
      suffix: ":12am, Case. Got a readout chipped into my optic nerve.'",
    },
  ],
  "02:45": [
    {
      time: "0245h",
      book: "Infinite Jest",
      author: "David Foster Wallace",
      prefix: "",
      suffix: "., Ennet House, the hours that are truly wee",
    },
  ],
  "02:46": [
    {
      time: "2.46am.",
      book: "Patriots",
      author: "Steve Sohmer",
      prefix: "",
      suffix:
        " The chain drive whirred and the paper target slid down the darkened range, ducking in and out of shafts of yellow incandescent light. At the firing station, a figure waited in the shadows. As the target passed the twenty-five-foot mark, the man opened fire: eight shots-rapid, unhesitating",
    },
    {
      time: "Two forty-six",
      book: "Blood Lines",
      author: "Tanya Huff",
      prefix: "Vicki shoved her glasses at her face and peered at the clock. ",
      suffix:
        ". 'I don't have time for this' she muttered, sttling back against the pillows, heart still slamming against her ribs.",
    },
  ],
  "02:47": [
    {
      time: "2.47am.",
      book: "The Book of Want",
      author: "Daniel A. Olivas",
      prefix: "The glowing numbers read ",
      suffix:
        " Moisés sighs and turns back to the bathroom door. Finally, the doorknob turns and Conchita comes back to bed. She resumes her place next to Moisés. Relieved, he pulls her close.",
    },
  ],
  "02:55": [
    {
      time: "2:55 a.m.",
      book: "The Night People",
      author: "Jack Finney",
      prefix:
        "\"It's the way the world will end, Harry. Recorded cocktail music nuclear-powered to play on for centuries after all life has been destroyed. Selections from 'No, No, Nanette,' throughout eternity. That do you for ",
      suffix: '?"',
    },
    {
      time: "2.55am.",
      book: "Downriver",
      author: "Iain Sinclair",
      prefix: "Time to go: ",
      suffix: " Two-handed, Cec lifted his peak cap from the chair.",
    },
  ],
  "02:56": [
    {
      time: "2:56",
      book: "Extremely Loud and Incredibly Close",
      author: "Jonathan Safran Foer",
      prefix: "It was ",
      suffix:
        " when the shovel touched the coffin. We all heard the sound and looked at each other.",
    },
  ],
  "02:59": [
    {
      time: "2.59",
      book: "The Ipcress File",
      author: "Len Deighton",
      prefix: "I remembered arriving in this room at ",
      suffix:
        ' one night. I remembered the sergeant who called me names: mostly Anglo-Saxon monosyllabic four-letter ones with an odd "Commie" thrown in for syntax.',
    },
  ],
  "03:00": [
    {
      time: "three o'clock",
      book: "The Voyage Out",
      author: "Virginia Woolf",
      prefix: '"She died this morning, very early, about ',
      suffix: '."',
    },
    {
      time: "Three in the morn.",
      book: "Something Wicked This Way Comes",
      author: "Ray Bradbury",
      prefix: "Three a.m. That’s our reward. ",
      suffix:
        " The soul’s midnight. The tide goes out, the soul ebbs. And a train arrives at an hour of despair. Why?",
    },
    {
      time: "three o'clock",
      book: "The Long Dark Tea-time of the Soul",
      author: "Douglas Adams",
      prefix: "According to her watch it was shortly after ",
      suffix: ", and according to everything else it was night-time.",
    },
    {
      time: "At three am",
      book: "The Long Goodbye",
      author: "Raymond Chandler",
      prefix: "",
      suffix:
        " I was walking the floor and listening to Katchaturian working in a tractor factory. He called it a violin concerto. I called it a loose fan belt and the hell with it",
    },
    {
      time: "three o' clock in the morning",
      book: "The Medusa Frequency",
      author: "Russell Hoban",
      prefix: "At ",
      suffix:
        " Eurydice is bound to come into it. After all, why did I sit here like a telegrapher at a lost outpost if not to receive messages from everywhere about the lost Eurydice who was never mine to begin with but whom I lamented and sought continually both professionally and amateurishly. This is not a digression. Where I am at three o' clock in the morning - and by now every hour is three o' clock in the morning - there are no digressions, it's all one thing.",
    },
    {
      time: "at three o’clock in the morning",
      book: "The Crack-Up",
      author: "F. Scott Fitzgerald",
      prefix: "But ",
      suffix:
        ", a forgotten package has the same tragic importance as a death sentence, and the cure doesn’t work -- and in a real dark night of the soul it is always three o’clock in the morning, day after day.",
    },
    {
      time: "three o'clock",
      book: "Afternoon Raag",
      author: "Amit Chaudhuri",
      prefix:
        "Early mornings, my mother is about, drifting in her pale nightie, making herself a cup of tea in the kitchen. Water begins to boil in the kettle; it starts as a private, secluded sound, pure as rain, and grows to a steady, solipsistic bubbling. Not till she has had one cup of tea, so weak that it has a colour accidentally golden, can she begin her day. She is an insomniac. Her nights are wide-eyed and excited with worry. Even at ",
      suffix:
        " in the morning one might hear her eating a Bain Marie biscuit in the kitchen.",
    },
    {
      time: "3 a.m.",
      book: "Songs from the Other Side of the Wall",
      author: "Dan Holloway",
      prefix:
        "I slam the phone down but it misses the base. I hit the clock instead, which flashes ",
      suffix: "",
    },
    {
      time: "3 o'clock",
      book: "The Crack-Up",
      author: "F. Scott Fitzgerald",
      prefix: "In a real dark night of the soul it is always ",
      suffix: " in the morning.",
    },
    {
      time: "at three A.M.",
      book: "Nemesis",
      author: "Philip Roth",
      prefix:
        "It was six untroubled days later – the best days at the camp so far, lavish July light thickly spread everywhere, six masterpiece mountain midsummer days, one replicating the other – that someone stumbled jerkily, as if his ankles were in chains, to the Comanche cabin’s bathroom ",
      suffix: "",
    },
    {
      time: "three in the morning",
      book: "Solar",
      author: "Ian McEwan",
      prefix: "It was ",
      suffix:
        " when his taxi stopped by giant mounds of snow outside his hotel. He had not eaten in hours.",
    },
    {
      time: "three o'clock at night",
      book: "I'm Not Stiller",
      author: "Max Frisch",
      prefix: "Once I saw a figure I shall never forget. It was ",
      suffix:
        ", as I was going home from Blacky as usual; it was a short-cut for me, and there would be nobody in the street at this time of night, I thought, especially not in this frightful cold.",
    },
    {
      time: "Three AM.",
      book: "The Game Players of Titan",
      author: "Philip K Dick",
      prefix:
        "Roused from her sleep, Freya Gaines groped for the switch of the vidphone; groggily she found it and snapped it on. 'Lo,' she mumbled, wondering what time it was. She made out the luminous dial of the clock beside the bed. ",
      suffix: " Good grief.",
    },
    {
      time: "0300",
      book: "Infinite Jest",
      author: "David Foster Wallace",
      prefix:
        "Schact clears his mouth and swallows mightily. 'Tavis can't even regrout tile in the locker room without calling a Community meeting or appointing a committee. The Regrouting Committee's been dragging along since may. Suddenly they're pulling secret ",
      suffix: " milk-switches? It doesn't ring true, Jim.",
    },
    {
      time: "Three in the morning",
      book: "Something Wicked This Way Comes",
      author: "Ray Bradbury",
      prefix: "",
      suffix:
        ", thought Charles Halloway, seated on the edge of his bed. Why did the train come at that hour? For, he thought, it’s a special hour. Women never wake then, do they? They sleep the sleep of babes and children. But men in middle age? They know that hour well",
    },
    {
      time: "three",
      book: "Three Men in a Boat",
      author: "Jerome K Jerome",
      prefix:
        'What\'s the time?" said the man, eyeing George up and down with evident suspicion; "why, if you listen you will hear it strike." George listened, and a neighbouring clock immediately obliged. "But it\'s only gone ',
      suffix: '!" said George in an injured tone, when it had finished.',
    },
    {
      time: "3:00 a.m.",
      book: "Desperate Characters",
      author: "Paula Fox",
      prefix: "When Sophie awoke, it was ",
      suffix: "",
    },
    {
      time: "three o’clock",
      book: "Middlemarch",
      author: "George Eliot",
      prefix: "You hearken, Missy. It’s ",
      suffix:
        " in the morning and I’ve got all my faculties as well as ever I had in my life. I know all my property and where the money’s put out. And I’ve made everything ready to change my mind, and do as I like at the last. Do you hear, Missy? I’ve got my faculties.”",
    },
  ],
  "03:01": [
    {
      time: "about three o'clock",
      book: "The Short Happy Life of Francis Macomber",
      author: "Ernest Hemingway",
      prefix: "It was now ",
      suffix:
        " in the morning and Francis Macomber, who had been asleep a little while after he had stopped thinking about the lion, wakened and then slept again.",
    },
  ],
  "03:04": [
    {
      time: "3.04",
      book: "The Return of the Dancing Master",
      author: "Henning Mankell",
      prefix:
        "…his back-up alarm clock rang. He looked at his front-line clock on the bedside table and noted that it had stopped at ",
      suffix: ". So, you couldn’t even rely on alarm clocks.",
    },
  ],
  "03:05": [
    {
      time: "3:05 a.m.",
      book: "The Corrections",
      author: "Jonathan Franzen",
      prefix: "On the Sunday before Christmas she awoke at ",
      suffix:
        " and though: Thirty-six hours. Four hours later she got up thinking: Thirty-two hours. Late in the day she took Alfred to the street-association Christmas party at Dale and Honey Driblett’s, sat him down safely with Kirby Root, and proceeded to remind all her neighbors that her favorite grandson, who’d been looking forward all year to a Christmas in St. Jude, was arriving tomorrow afternoon.",
    },
  ],
  "03:07": [
    {
      time: "3.07am",
      book: "The Cold Six Thousand",
      author: "James Ellroy",
      prefix: "Wayne late-logged in: ",
      suffix:
        " -the late-late show. He parked. He dumped his milk can. He yawned, he stretched. He scratched.",
    },
  ],
  "03:10": [
    {
      time: "ten-past three",
      book: "The Whole Story and Other Stories",
      author: "Ali Smith",
      prefix:
        "I think my credit card was in there too. I wrote down the words credit card and said that if they wouldn't let me cancel them I'd demand that they registered the loss so you couldn't be charge for anything beyond the time of my calling them up. I looked at the clock. It was ",
      suffix: ".",
    },
    {
      time: "ten past three",
      book: "Love Again",
      author: "Philip Larkin",
      prefix: "Love again; wanking at ",
      suffix: "",
    },
  ],
  "03:14": [
    {
      time: "3.14",
      book: "The Slap",
      author: "Christos Tsiolkas",
      prefix:
        "Since he had told the girl that it had to end, he'd been waking up every morning at ",
      suffix:
        ", without fail. Every morning his eyes would flick open, alert, and the red numerals on his electric alarm clock would read 3.14.",
    },
  ],
  "03:15": [
    {
      time: "3:15",
      book: "Manhole 69",
      author: "JG Ballard",
      prefix: "Above the door of Room 69 the clock ticked on at ",
      suffix:
        ". The motion was accelerating. What had once been the gymnasium was now a small room, seven feet wide, a tight, almost perfect cube.",
    },
  ],
  "03:17": [
    {
      time: "3:17",
      book: "The Goldfinch",
      author: "Donna Tartt",
      prefix:
        "The two of us sat there, listening—Boris more intently than me. “Who’s that with him then?” I said. “Some whore.” He listened for a moment, brow furrowed, his profile sharp in the moonlight, and then lay back down. “Two of them.” I rolled over, and checked my iPod. It was ",
      suffix: " in the morning.",
    },
    {
      time: "3.17 a.m.",
      book: "What Was Lost",
      author: "Catherine O'Flynn",
      prefix:
        "He turned to the monitors again and flicked through the screens, each one able to display eight different camera mountings, giving Kurt 192 different still lives of Green Oaks at ",
      suffix: " this March night.",
    },
  ],
  "03:19": [
    {
      time: "3.19 A.M.",
      book: "House of Leaves",
      author: "Mark Z Danielewski",
      prefix:
        "The time stamp on Navidson's camcorder indicates that it is exactly ",
      suffix: "",
    },
  ],
  "03:20": [
    {
      time: "3.20am",
      book: "Anil's Ghost",
      author: "Michael Ondaatje",
      prefix: "Prabath Kumara, 16. 17th November 1989. At ",
      suffix: " from the home of a friend.",
    },
  ],
  "03:21": [
    {
      time: "twenty-one minutes past three",
      book: "The Toilers of the Sea",
      author: "Victor Hugo",
      prefix:
        "Next, he remembered that the morrow of Christmas would be the twenty-seventh day of the moon, and that consequently high water would be at ",
      suffix:
        ", the half-ebb at a quarter past seven, low water at thirty-three minutes past nine, and half flood at thirty-nine minutes past twelve.",
    },
  ],
  "03:25": [
    {
      time: "3:25 a.m.",
      book: "We Were the Mulvaneys",
      author: "Joyce Carol Oates",
      prefix: "It was ",
      suffix:
        " A strange thrill, to think I was the only Mulvaney awake in the house.",
    },
  ],
  "03:28": [
    {
      time: "3.28",
      book: "Dreams of Leaving",
      author: "Rupert Thomson",
      prefix:
        "Now somebody was running past his room. A door slammed. That foreign language again. What the devil was going on? he switched on his light and peered at his watch. ",
      suffix: ". He got out of bed.",
    },
  ],
  "03:30": [
    {
      time: "Half past Three",
      book: "At Half past Three, a single Bird",
      author: "Emily Dickinson",
      prefix: "At ",
      suffix:
        ", a single Bird Unto a silent Sky Propounded but a single term Of cautious melody.",
    },
    {
      time: "half-past three A.M.",
      book: "Les Miserables",
      author: "Victor Hugo",
      prefix: "At ",
      suffix:
        " he lost one illusion: officers sent to reconnoitre informed him that the enemy was making no movement.",
    },
    {
      time: "3:30 A.M.",
      book: "The World According to Garp",
      author: "John Irving",
      prefix: "It's ",
      suffix:
        " in Mrs. Ralph's finally quiet house when Garp decides to clean the kitchen, to kill the time until dawn. Familiar with a housewife's tasks, Garp fills the sink and starts to wash the dishes.",
    },
    {
      time: "three-thirty",
      book: "Whoever Was Using This Bed",
      author: "Raymond Carver",
      prefix:
        'Let\'s go to sleep, I say. "Look at what time it is." The clock radio is right there beside the bed. Anyone can see it says ',
      suffix: ".",
    },
    {
      time: "three thirty",
      book: "How to Eat Fried Worms",
      author: "Thomas Rockwell",
      prefix: "Now, look. I am not going to call Dr. McGrath at ",
      suffix:
        " in the morning to ask if it's all right for my son to eat worms. That's flat.",
    },
  ],
  "03:33": [
    {
      time: "3:33",
      book: "Chronic City",
      author: "Jonathan Lethem",
      prefix:
        "A draft whistled in around the kitchen window frame and I shivered. The digital clock on Perkus's stove read ",
      suffix: ".",
    },
  ],
  "03:34": [
    {
      time: "3:34 am.",
      book: "Always Florence",
      author: "Muriel Jensen",
      prefix: "It was ",
      suffix:
        " and he was wide-awake. He'd heard the phone ring and the sound of his uncle's voice.",
    },
  ],
  "03:35": [
    {
      time: "3.35 a.m.",
      book: "The Dogs of Riga",
      author: "Henning Mankell",
      prefix:
        "He could just see the hands of the alarm clock in the darkness: ",
      suffix: " He adjusted his pillow and shut his eyes.",
    },
  ],
  "03:36": [
    {
      time: "3:36 a.m.",
      book: "Zoopraxis",
      author: "Richard C Matheson",
      prefix: "As I near Deadhorse, it's ",
      suffix:
        " and seventeen below. Tall, sodium vapor lights spill on the road and there are no trees, only machines, mechanical shadows. There isn't even a church. It tells you everything.",
    },
  ],
  "03:37": [
    {
      time: "thirty-seven A.M.",
      book: "The Cobweb",
      author: "Stephen Bury",
      prefix: "It was three ",
      suffix:
        ", and for once Maggie was asleep. She had got to be a pretty good sleeper in the last few months. Clyde was prouder of this fact than anything.",
    },
  ],
  "03:38": [
    {
      time: "3.38am",
      book: "Just Like the Ones we Used to Know",
      author: "Connie Willis",
      prefix: "At ",
      suffix:
        ", it began to snow in Bowling Green, Kentucky. The geese circling the city flew back to the park, landed, and hunkered down to sit it out on their island in the lake.",
    },
  ],
  "03:39": [
    {
      time: "3.39am.",
      book: "The Clockwork man",
      author: "William Jablonsky",
      prefix: "23 October 1893 ",
      suffix:
        " Upon further thought, I feel it necessary to explain that exile into the Master's workshop is not an unpleasant fate. It is not simply some bare-walled cellar devoid of stimulation - quite the opposite.",
    },
  ],
  "03:40": [
    {
      time: "three forty",
      book: "Saturday",
      author: "Ian McEwan",
      prefix: "His bedside clock shows ",
      suffix:
        ". He has no idea what he's doing out of bed: he has no need to relieve himself, nor is he disturbed by a dream or some element of the day before, or even by the state of the world.",
    },
  ],
  "03:41": [
    {
      time: "3.41am.",
      book: "The Dogs of Riga",
      author: "Henning Mankell",
      prefix: "The alarm clock said ",
      suffix:
        " He sat up. Why was the alarm clock slow? He picked up the alarm clock and adjusted the hands to show the same time as his wristwatch: 3.44am",
    },
  ],
  "03:42": [
    {
      time: "3:42",
      book: "Bride Comes to Yellow Sky",
      author: "Stephen Crane",
      prefix: '"We are due in Yellow Sky at ',
      suffix:
        '," he said, looking tenderly into her eyes. ""Oh, are we?"" she said, as if she had not been aware of it. To evince surprise at her husband\'s statement was part of her wifely amiability.',
    },
  ],
  "03:43": [
    {
      time: "3.43am.",
      book: "Ghostwritten",
      author: "David Mitchell",
      prefix: "The clock says ",
      suffix:
        " The thermometer says it's a chilly fourteen degrees Fahrenheit. The weatherman says the cold spell will last until Thursday, so bundle up and bundle up some more. There are icicles barring the window of the bat cave.",
    },
  ],
  "03:44": [
    {
      time: "3.44 a.m.",
      book: "Liver: Leberknödel",
      author: "Will Self",
      prefix:
        "It was dark. After she had switched the light on and been to the toilet, she checked her watch: ",
      suffix:
        " She undressed, put the cat out the door and returned to the twin bed.",
    },
  ],
  "03:45": [
    {
      time: "quarter to four",
      book: "An Ideal Husband",
      author: "Oscar Wilde",
      prefix:
        "LORD CAVERSHAM: Well, sir! what are you doing here? Wasting your life as usual! You should be in bed, sir. You keep too late hours! I heard of you the other night at Lady Rufford's dancing till four o' clock in the morning! LORD GORING: Only a ",
      suffix: ", father.",
    },
  ],
  "03:47": [
    {
      time: "3:47",
      book: "The Curious Incident Of The Dog In The Night-Time",
      author: "Mark Haddon",
      prefix: "I stayed awake until ",
      suffix:
        ". That was the last time I looked at my watch before I fell asleep. It has a luminous face and lights up if you press a button so I could read it in the dark. I was cold and I was frightened Father might come out and find me. But I felt safer in the garden because I was hidden.",
    },
  ],
  "03:49": [
    {
      time: "3.49",
      book: "The Ipcress File",
      author: "Len Deighton",
      prefix: "It was ",
      suffix:
        ' when he hit me because of the two hundred times I had said, "I don\'t know." He hit me a lot after that.',
    },
  ],
  "03:50": [
    {
      time: "ten or five to four",
      book: "A Death in Brazil: A Book of Omissions",
      author: "Peter Robb",
      prefix:
        "She had used her cell phone to leave several messages on the answering machine in Sao Paulo of the young dentist of the previous evening, whose name was Fernando. The first was recorded at ",
      suffix:
        " in the morning. I'm never going to forget you ... I'm sure we'll meet again somewhere.",
    },
  ],
  "03:51": [
    {
      time: "3:51",
      book: "White Noise",
      author: "Don DeLillo",
      prefix:
        "I lacked the will and physical strength to get out of bed and move through the dark house, clutching walls and stair rails. To feel my way, reinhabit my body, re-enter the world. Sweat trickled down my ribs. The digital reading on the clock-radio was ",
      suffix:
        ". Always odd numbered at times like this. What does it mean? Is death odd-numbered?",
    },
    {
      time: "3:51",
      book: "White Noise",
      author: "Don DeLillo",
      prefix: "The digital reading on the clock-radio was ",
      suffix:
        ". Always odd numbers at times like this. What does it mean? Is death odd-numbered?",
    },
  ],
  "03:54": [
    {
      time: "3.54 a.m.",
      book: "The More a Man Has, the More a Man Wants",
      author: "Paul Muldoon",
      prefix:
        "The charter flight from Florida touched down at Aldergrove minutes earlier, at ",
      suffix: "",
    },
  ],
  "03:55": [
    {
      time: "3.55 a.m.",
      book: "Saturday",
      author: "Ian McEwan",
      prefix: "Here in the cavernous basement at ",
      suffix: ", in a single pool of light, is Theo Perowne.",
    },
  ],
  "03:57": [
    {
      time: "Nearly four",
      book: "Sometimes a Great Notion",
      author: "Ken Kesey",
      prefix:
        'Certain facts were apparent: dark; cold; thundering boots; quilts; pillow; light under the door – the materials of reality - but I could not pin these materials down in time. And the raw materials of reality without that glue of time are materials adrift and reality is as meaningless as the balsa parts of a model airplane scattered to the wind...I am in my old room, yes, in the dark, certainly, and it is cold, obviously, but what time is it? "',
      suffix: ', son." But I mean what time?',
    },
  ],
  "03:58": [
    {
      time: "two minutes to four",
      book: "Heartland",
      author: "Wilson Harris",
      prefix:
        "The ancient house was deserted, the crumbling garage padlocked, and one was just able to discern - by peering through a crack in the bubbling sun on the window - the face of a clock on the opposite wall. The clock had stopped at ",
      suffix:
        " early in the morning, or who could tell, it may have been earlier still, yesterday in the afternoon, a couple of hours after Kaiser had left Kamaria for Bartica.",
    },
    {
      time: "3:58",
      book: "Underworld",
      author: "Don Delillo",
      prefix: "The clock atop the clubhouse reads ",
      suffix: ".",
    },
  ],
  "03:59": [
    {
      time: "Nearly four",
      book: "Sometimes a Great Notion",
      author: "Ken Kesey",
      prefix:
        'And the raw materials of reality without that glue of time are materials adrift and reality is as meaningless as the balsa parts of a model airplane scattered to the wind...I am in my old room, yes, in the dark, certainly, and it is cold, obviously, but what time is it? "',
      suffix: ', son."',
    },
  ],
  "04:00": [
    {
      time: "four o'clock",
      book: "The Great Gatsby",
      author: "F. Scott Fitzgerald",
      prefix: '"Nothing happened," he said wanly. "I waited, and about ',
      suffix:
        ' she came to the window and stood there for a minute and then turned out the light."',
    },
    {
      time: "four am.",
      book: "Watermelon",
      author: "Marian Keyes",
      prefix: "I looked at the clock and it was (yes, you guessed it) ",
      suffix:
        ' I should have taken comfort from the fact that approximately quarter of the Greenwich Mean Time world had just jolted awake also and were lying, staring miserably into the darkness, worrying ..."',
    },
    {
      time: "4am",
      book: "Atomised",
      author: "Michel Houellebecq",
      prefix:
        "Suddenly, he started to cry. Curled up on the sofa he sobbed loudly. Michel looked at his watch; it was just after ",
      suffix: ". On the screen a wild cat had a rabbit in its mouth.",
    },
    {
      time: "Four o'clock",
      book: "The Birds begun at Four o'clock",
      author: "Emily Dickinson",
      prefix: "The Birds begun at ",
      suffix: "— Their period for Dawn—",
    },
    {
      time: "at four",
      book: "2666",
      author: "Roberto Bolano",
      prefix: "The night before Albert Kessler arrived in Santa Teresa, ",
      suffix:
        " in the morning, Sergio Gonzalez Rodriguez got a call from Azucena Esquivel Plata, reporter and PRI congresswoman.",
    },
    {
      time: "at four",
      book: "Aubade",
      author: "Philip Larkin",
      prefix: "Waking ",
      suffix:
        " to soundless dark, I stare. In time the curtain-edges will grow light. Till then I see what's really always there: Unresting death, a whole day nearer now, Making all thought impossible but how And where and when I shall myself die.",
    },
    {
      time: "at four",
      book: "The Tea Rose",
      author: "Jennifer Donnelly",
      prefix:
        "When he noticed that the chefs from the grand hotels and restaurants - a picky, impatient bunch - tended to move around from seller to seller, buying apples here and broccoli there, he asked if he could have tea available for them. Tommy agreed, and the chefs, grateful for a hot drink ",
      suffix: " in the morning, lingered and bought.",
    },
  ],
  "04:01": [
    {
      time: "just after 4am",
      book: "Atomised",
      author: "Michel Houellebecq",
      prefix:
        "Suddenly, he started to cry. Curled up on the sofa he sobbed loudly. Michel looked at his watch; it was ",
      suffix: ". On the screen a wild cat had a rabbit in its mouth.",
    },
  ],
  "04:02": [
    {
      time: "4:02",
      book: "The History of Love",
      author: "Nicole Krauss",
      prefix:
        "I walked up and down the row. No one gave me a second look. Finally I sat down next to a man. He paid no attention. My watch said ",
      suffix: ". Maybe he was late.",
    },
  ],
  "04:03": [
    {
      time: "4:03 a.m.",
      book: "The Time Traveler's Wife",
      author: "Audrey Niffenegger",
      prefix: "It's ",
      suffix:
        " on a supremely cold January morning and I'm just getting home. I've been out dancing and I'm only half drunk but utterly exhausted.",
    },
  ],
  "04:04": [
    {
      time: "Four minutes after four!",
      book: "Angel Hill",
      author: "Cirilo Villaverde",
      prefix: "",
      suffix:
        " It's still very early and to get from here to there won't take me more than 15 minutes, even walking slowly. She told me around five o'clock. Wouldn't it be better to wait on the corner",
    },
  ],
  "04:05": [
    {
      time: "4.05am.",
      book: "We Were the Mulvaneys",
      author: "Joyce Carol Oates",
      prefix: "Leaves were being blown against my window. It was ",
      suffix:
        " The moon had shifted in the sky, glaring through a clotted mass of clouds like a candled egg.",
    },
  ],
  "04:06": [
    {
      time: "4.06am",
      book: "The Expats",
      author: "Chris Pavone",
      prefix:
        "Dexter looked at Kate's note, then her face, then the clock. It was ",
      suffix: ", the night before they would go to the restaurant.",
    },
  ],
  "04:07": [
    {
      time: "4.07am.",
      book: "Guarding Hanna: A Novel",
      author: "Miha Mazzini",
      prefix: "",
      suffix:
        " Why am I standing? My shoulders feel cold and I'm shivering. I become aware that I'm standing in the middle of the room. I immediately look at the bedroom door. Closed, with no signs of a break-in. Why did I get up",
    },
  ],
  "04:08": [
    {
      time: "4:08 a.m.",
      book: "Dying in the Twilight of Summer",
      author: "Seth O'Connell",
      prefix: "It was at ",
      suffix:
        " beneath the cool metal of a jungle gym that all Andrew's dreams came true. He kissed his one true love and swore up and down that it would last forever to this exhausted companion throughout their long trek home.",
    },
  ],
  "04:11": [
    {
      time: "eleven minutes after four",
      book: "The Stuff of Life",
      author: "Karen Karbo",
      prefix: "The next morning I awaken at exactly ",
      suffix:
        ", having slept straight through my normal middle-of-the-night insomniac waking at three.",
    },
  ],
  "04:12": [
    {
      time: "four-twelve",
      book: "Hard Boiled Wonderland and the End of the World",
      author: "Haruki Murakami",
      prefix:
        "Finally, she signalled with her light that she'd made it to the top. I signalled back, then shined the light downward to see how far the water had risen. I couldn't make out a thing. My watch read ",
      suffix:
        " in the morning. Not yet dawn. The morning papers still not delivered, trains not yet running, citizens of the surface world fast asleep, oblivious to all this. I pulled the rope taut with both hands, took a deep breath, then slowly began my climb.",
    },
    {
      time: "4:12",
      book: "Get Shorty",
      author: "Elmore Leonard",
      prefix:
        "Karen felt the bed move beneath Harry's weight. Lying on her side she opened her eyes to see digital numbers in the dark, ",
      suffix:
        " in pale green. Behind her Harry continued to move, settling in. She watched the numbers change to 4:13.",
    },
  ],
  "04:13": [
    {
      time: "4:13",
      book: "Get Shorty",
      author: "Elmore Leonard",
      prefix:
        "Karen felt the bed move beneath Harry's weight. Lying on her side she opened her eyes to see digital numbers in the dark, 4:12 in pale green. Behind her Harry continued to move, settling in. She watched the numbers change to ",
      suffix: ".",
    },
  ],
  "04:14": [
    {
      time: "4:14 a.m.",
      book: "A Real Nightmare",
      author: "David H Swendsen",
      prefix: "At ",
      suffix:
        ", the two men returned to the Jeep. After the passenger replaced the cans in the back of the Jeep, the driver backed out of the driveway and headed east. The last images found on the film appeared to be flames or smoke.",
    },
  ],
  "04:15": [
    {
      time: "four-fifteen",
      book: "Pigs in Heaven",
      author: "Barbara Kingsolver",
      prefix:
        "Alice wants to warn her that a defect runs in the family, like flat feet or diabetes: they're all in danger of ending up alone by their own stubborn choice. The ugly kitchen clock says ",
      suffix: ".",
    },
  ],
  "04:16": [
    {
      time: "Four-sixteen",
      book: "Hard Boiled Wonderland and the End of the World",
      author: "Haruki Murakami",
      prefix: "I stooped to pick up my watch from the floor. ",
      suffix:
        ". Another hour until dawn. I went to the telephone and dialled my own number. It'd been a long time since I'd called home, so I had to struggle to remember the number. I let it ring fifteen times; no answer. I hung up, dialled again, and let it ring another fifteen times. Nobody.",
    },
    {
      time: "four sixteen",
      book: "Freaks in the City: Book Two of the Freaks Series",
      author: "Maree Anderson",
      prefix: "They pulled into the visitor's carpark at ",
      suffix:
        " am. He knew it was four sixteen because the entrance to the maternity unit sported a digital clock beneath the signage.",
    },
  ],
  "04:17": [
    {
      time: "4.17am",
      book: "The Vile",
      author: "Douglas Phinney",
      prefix: "He awoke at ",
      suffix:
        " in a sweat. He had been dreaming of Africa again, and then the dream had continued in the U.S. when he was a young man. But Inbata had been there, watching him.",
    },
  ],
  "04:18": [
    {
      time: "four-eighteen",
      book: "Hard Boiled Wonderland and the End of the World",
      author: "Haruki Murakami",
      prefix:
        "I grabbed the alarm clock, threw it on my lap, and slapped the red and black buttons with both hands. The ringing didn't stop. The telephone! The clock read ",
      suffix:
        '. It was dark outside. Four-eighteen a.m. I got out of bed and picked up the receiver. "Hello?"',
    },
  ],
  "04:22": [
    {
      time: "4.22",
      book: "The Ipcress File",
      author: "Len Deighton",
      prefix:
        "He hurt me to the point where I wanted to tell him something. My watch said ",
      suffix: " now. It had stopped. It was smashed.",
    },
  ],
  "04:23": [
    {
      time: "4:23",
      book: "Let The Right One In",
      author: "John Ajvide Lindqvist",
      prefix: "",
      suffix:
        ", Monday morning, Iceland Square. A number of people in the vicinity of Bjornsongatan are awakened by loud screams",
    },
    {
      time: "04:23",
      book: "Neuromancer",
      author: "William Gibson",
      prefix: "Her chip pulsed the time. ",
      suffix: ":04. It had been a long day.",
    },
  ],
  "04:25": [
    {
      time: "twenty-five minutes past four",
      book: "The Adventures of Sherlock Holmes",
      author: "Sir Arthur Conan Doyle",
      prefix:
        "As I dressed I glanced at my watch. It was no wonder that no one was stirring. It was ",
      suffix: ".",
    },
  ],
  "04:30": [
    {
      time: "four thirty",
      book: "Essays on Love",
      author: "Alain de Botton",
      prefix:
        "At the end of a relationship, it is the one who is not in love who makes the tender speeches. I was overwhelmed by a sense of betrayal, betrayal because a union in which I had invested so much had been declared bankrupt without my feeling it to be so. Chloe had not given it a chance, I argued with myself, knowing the hopelessness of these inner courts announcing hollow verdicts at ",
      suffix: " in the morning.",
    },
    {
      time: "0430",
      book: "Infinite Jest",
      author: "David Foster Wallace",
      prefix:
        "Hester Thrale undulates in in a false fox jacket at 2330 as usual even though she has to be up at like ",
      suffix:
        " for the breakfast shift at the Provident Nursing Home and sometimes eats breakfast with Gately, both their faces nodding perilously close to their Frosted Flakes.",
    },
    {
      time: "0430",
      book: "Infinite Jest",
      author: "David Foster Wallace",
      prefix:
        "Tonight Clenette H. and the deeply whacked out Yolanda W. come back in from Footprints around 2315 in purple skirts and purple lipstick and ironed hair, tottering on heels and telling each other what a wicked time they just had. Hester Thrale undulates in in a false fox jacket at 2330 as usual even though she has to be up at like ",
      suffix:
        " for the breakfast shift at the Provident Nursing Home and sometimes eats breakfast with Gately, both their faces nodding perilously close to their Frosted Flakes.",
    },
  ],
  "04:31": [
    {
      time: "4:31",
      book: "Microserfs",
      author: "Douglas Coupland",
      prefix: "An earthquake hit Los Angeles at ",
      suffix: " this morning and the images began arriving via CNN right away.",
    },
  ],
  "04:32": [
    {
      time: "4:32 a.m.",
      book: "Nineteen Minutes",
      author: "Jodi Picoult",
      prefix: "On his first day of kindergarten, Peter Houghton woke up at ",
      suffix:
        " He padded into his parents' room and asked if it was time yet to take the school bus.",
    },
  ],
  "04:35": [
    {
      time: "4:35",
      book: "Dreams and Shadows",
      author: "C Robert Cargill",
      prefix:
        "No manner of exhaustion can keep a child asleep much later than six a.m. on Christmas Day. Colby awoke at ",
      suffix: ".",
    },
  ],
  "04:36": [
    {
      time: "4:36",
      book: "The Brass Go-Between",
      author: "Ross Thomas",
      prefix: "At ",
      suffix:
        " that morning, alone in my hotel room, it had been a much better scene. Spencer had blanched, confounded by the inescapable logic of my accusation. A few drops of perspiration had formed on his upper lip. A tiny vein had started to throb in his temple.",
    },
  ],
  "04:38": [
    {
      time: "4.38 a.m.",
      book: "The Queue",
      author: "Jonathan Barrow",
      prefix: "At ",
      suffix:
        " as the sun is coming up over Gorley Woods, I hear a strange rustling in the grass beside me. I peer closely but can see nothing.",
    },
  ],
  "04:40": [
    {
      time: "4.40am",
      book: "Bossypants",
      author: "Tina Fey",
      prefix: "I settled into a daily routine. Wake up at ",
      suffix: ", shower, get on the train north by ten after five.",
    },
  ],
  "04:41": [
    {
      time: "4:41",
      book: "Damaged Goods: A Novel",
      author: "Roland S. Jefferson",
      prefix: "At ",
      suffix:
        " Crane's voice crackled through the walkie-talkie as if he'd read their thoughts of mutiny. “Everyone into the elevator. Now!” Only moments before the call he and C.J. had finished what they hoped would be a successful diversion.",
    },
  ],
  "04:43": [
    {
      time: "four forty-three",
      book: "Pyschoraag",
      author: "Suhayl Saadi",
      prefix: "The time is ",
      suffix: " in the mornin an it's almost light oot there.",
    },
  ],
  "04:45": [
    {
      time: "4:45 a.m.",
      book: "Faceless Killers",
      author: "Henning Mankell",
      prefix:
        "He lies still in the darkness and listens. His wife's breathing at his side is so faint that he can scarcely hear it. One of these mornings she'll be lying dead beside me and I won't even notice, he thinks. Or maybe it'll be me. Daybreak will reveal that one of us has been left alone. He checks the clock on the table next to the bed. The hands glow and register ",
      suffix: "",
    },
    {
      time: "4:45 a.m.",
      book: "Faceless Killers",
      author: "Henning Mankell",
      prefix:
        "His wife's breathing at his side is so faint that he can scarcely hear it. One of these mornings she'll be lying dead beside me and I won't even notice, he thinks. Or maybe it'll be me. Daybreak will reveal that one of us has been left alone. He checks the clock on the table next to the bed. The hands glow and register ",
      suffix: "",
    },
  ],
  "04:46": [
    {
      time: "four-forty-six",
      book: "Hard Boiled Wonderland and the End of the World",
      author: "Haruki Murakami",
      prefix: "The phone rang again at ",
      suffix:
        '."Hello," I said. "Hello," came a woman\'s voice. "Sorry about the time before. There\'s a disturbance in the sound field. Sometimes the sound goes away." "The sound goes away?" "Yes," she said. "The sound field\'s slipping. Can you hear me?" "Loud and clear," I said. It was the granddaughter of that kooky old scientist who\'d given me the unicorn skull. The girl in the pink suit.',
    },
  ],
  "04:48": [
    {
      time: "4:48",
      book: "4:48 Psychosis",
      author: "Sarah Kane",
      prefix: "At ",
      suffix:
        " the happy hour when clarity visits warm darkness which soaks my eyes I know no sin",
    },
  ],
  "04:50": [
    {
      time: "ten minutes to five",
      book: "The 13 Clocks",
      author: "James Thurber",
      prefix:
        "Even the hands of his watch and the hands of all the thirteen clocks were frozen. They had all frozen at the same time, on a snowy night, seven years before, and after that it was always ",
      suffix: " in the castle.",
    },
  ],
  "04:54": [
    {
      time: "Six minutes to five",
      book: "Dawn: A Novel",
      author: "Elie Wiesel",
      prefix: "",
      suffix:
        ". Six minutes to go. Suddenly I felt quite clearheaded. There was an unexpected light in the cell; the boundaries were drawn, the roles well defined. The time of doubt and questioning and uncertainty was over",
    },
  ],
  "04:55": [
    {
      time: "4:55",
      book: "Fear and Loathing: On the Campaign Trail '72",
      author: "Hunter S. Thompson",
      prefix: "",
      suffix:
        " - Mank holding phone. Turns to Caddell - 'Who is this?' Caddell: 'Jim.' (shrugs) 'I think he's our man in Cincinnati.",
    },
  ],
  "04:57": [
    {
      time: "few minutes before five",
      book: "A Death in Brazil: A Book of Omissions",
      author: "Peter Robb",
      prefix: "The second said the same thing a ",
      suffix:
        ", and mentioned eternity... I'm sure I'll meet you in the other world. Four minutes later she left a last, fleeting message: My love. Fernando. It's Suzana. Then, it seemed, she had shot herself.",
    },
  ],
  "04:58": [
    {
      time: "Two minutes to five",
      book: "Dawn: A Novel",
      author: "Elie Wiesel",
      prefix: "He wants to look death in the face. ",
      suffix:
        ". I took a handkerchief out of my pocket, but John Dawson ordered me to put it back. An Englishman dies with his eyes open. He wants to look death in the face.",
    },
  ],
  "04:59": [
    {
      time: "0459",
      book: "Infinite Jest",
      author: "David Foster Wallace",
      prefix:
        "The whole place smells like death no matter what the fuck you do. Gately gets to the shelter at ",
      suffix:
        ".9h and just shuts his head off as if his head had a control switch.",
    },
  ],
  "05:00": [
    {
      time: "Five o'clock",
      book: "Jane Eyre",
      author: "Charlotte Brontë",
      prefix: "",
      suffix:
        " had hardly struck on the morning of the 19th of January, when Bessie brought a candle into my closet and found me already up and nearly dressed",
    },
    {
      time: "Five o'clock",
      book: "Jane Eyre",
      author: "Charlotte Brontë",
      prefix: "",
      suffix:
        " had hardly struck on the morning of the 19th of January, when Bessie brought a candle into my closet and found me already up and nearly dressed. I had risen half-an-hour before her entrance, and had washed my face, and put on my clothes by the light of a half-moon just setting, whose rays streamed through the narrow window near my crib",
    },
    {
      time: "5 a.m.",
      book: "The Dunwich Horror",
      author: "H.P. Lovecraft",
      prefix:
        "It was in the township of Dunwich, in a large and hardly inhabited farmhouse set against a hillside 4 miles from the village and a mile and a half from any other dwelling, that Wilbur Whately was born at ",
      suffix:
        " on Sunday, 2 February, 1913. The date was recalled because it was Candlemas, which people in Dunwich curiously observe under another name...",
    },
    {
      time: "five o'clock",
      book: "Music and Silence",
      author: "Rose Tremain",
      prefix: "Just after ",
      suffix:
        " on this chill September morning, the fishmonger's cart, containing Kirsten and Emilia and such possessions as they have been able to assemble in the time allowed to them, is driven out of the gates of Rosenborg?",
    },
    {
      time: "Five",
      book: "The 13 Clocks",
      author: "James Thurber",
      prefix:
        'The cold eye of the Duke was dazzled by the gleaming of a thousand jewels that sparkled on the table. His ears were filled with chiming as the clocks began to strike. "One!" said Hark. "Two!" cried Zorn of Zorna. "Three!" the Duke\'s voice almost whispered. \'Four!" sighed Saralinda. "',
      suffix:
        '!" the Golux crowed, and pointed at the table. "The task is done, the terms are met," he said.',
    },
    {
      time: "five o'clock",
      book: "The Day Came Slow, Till Five O' Clock",
      author: "Emily Dickinson",
      prefix: "The day came slow, till ",
      suffix:
        ". Then sprang before the hills. Like hindered rubies, or the light. A sudden musket spills",
    },
    {
      time: "5 a.m.",
      book: "Things",
      author: "Fleur Adcock",
      prefix:
        "There are worse things than having behaved foolishly in public. There are worse things than these miniature betrayals, committed or endured or suspected; there are worse things than not being able to sleep for thinking about them. It is ",
      suffix:
        " All the worse things come stalking in and stand icily about the bed looking worse and worse and worse.",
    },
    {
      time: "five o'clock",
      book: "Vanity Fair",
      author: "William Makepeace Thackeray",
      prefix:
        'What causes young people to "come out," but the noble ambition of matrimony? What sends them trooping to watering-places? What keeps them dancing till ',
      suffix: " in the morning through a whole mortal season?",
    },
  ],
  "05:01": [
    {
      time: "one minute past five",
      book: "The clocks",
      author: "Agatha Christie",
      prefix: '"Oh yes. His clocks were set at ',
      suffix:
        ', four minutes past five and seven minutes past five. That was the combination number of a safe, 515457. The safe was concealed behind a reproduction of the Mona Lisa. Inside the safe," continued Poirot, with distaste, "were the Crown Jewels of the Russian Royal Family."',
    },
    {
      time: "after five o'clock",
      book: "Music and Silence",
      author: "Rose Tremain",
      prefix: "Just ",
      suffix:
        " on this chill September morning, the fishmonger's cart, containing Kirsten and Emilia and such possessions as they have been able to assemble in the time allowed to them, is driven out of the gates of Rosenborg?",
    },
  ],
  "05:02": [
    {
      time: "5:02 a.m.",
      book: "The Prize",
      author: "Brenda Joyce",
      prefix: "It was ",
      suffix:
        ", December 14. In another fifty-eight minutes he would set sail for America. He did not want to leave his bride; he did not want to go.",
    },
  ],
  "05:03": [
    {
      time: "5:03 a.m.",
      book: "Unhallowed ground",
      author: "Heather Graham",
      prefix: "It was ",
      suffix:
        " It didn't matter. She wasn't going to get back to sleep. She threw off her covers and, swearing at herself, Caleb and Mr. Griffin, she headed into the shower.",
    },
  ],
  "05:04": [
    {
      time: "four minutes past five",
      book: "The clocks",
      author: "Agatha Christie",
      prefix: '"Oh yes. His clocks were set at one minute past five, ',
      suffix:
        ' and seven minutes past five. That was the combination number of a safe, 515457. The safe was concealed behind a reproduction of the Mona Lisa. Inside the safe," continued Poirot, with distaste, "were the Crown Jewels of the Russian Royal Family."',
    },
    {
      time: "5.04 a.m.",
      book: "The Accidental",
      author: "Ali Smith",
      prefix: "",
      suffix:
        " on the substandard clock radio. Because why do people always say the day starts now? Really it starts in the middle of the night at a fraction of a second past midnight",
    },
    {
      time: "four minutes past five",
      book: "The Clocks",
      author: "Agatha Christie",
      prefix: "Oh yes. His clocks were set at one minute past five, ",
      suffix:
        ' and seven minutes past five. That was the combination number of a safe, 515457. The safe was concealed behind a reproduction of the Mona Lisa. Inside the safe, continued Poirot, with distaste, "were the Crown Jewels of the Russian Royal Family."',
    },
  ],
  "05:05": [
    {
      time: "five past five",
      book: "The Namesake",
      author: "Jhumpa Lahiri",
      prefix: "The baby, a boy, is born at ",
      suffix: " in the morning.",
    },
  ],
  "05:06": [
    {
      time: "5:06 a.m.",
      book: "This is Where I Leave you",
      author: "Jonathon Tropper",
      prefix: "",
      suffix:
        " I wake up strangely energized, my stomach growling. Upstairs, the overstocked fridge offers me its bounty of sympathy food",
    },
  ],
  "05:07": [
    {
      time: "seven minutes past five",
      book: "The clocks",
      author: "Agatha Christie",
      prefix:
        '"Oh yes. His clocks were set at one minute past five, four minutes past five and ',
      suffix:
        '. That was the combination number of a safe, 515457. The safe was concealed behind a reproduction of the Mona Lisa. Inside the safe," continued Poirot, with distaste, "were the Crown Jewels of the Russian Royal Family."',
    },
  ],
  "05:08": [
    {
      time: "5:08",
      book: "Letters",
      author: "John Barth",
      prefix: "Ambrose and I will marry at Fort McHenry at ",
      suffix: " EDST this coming Saturday, Rosh Hashanah!",
    },
  ],
  "05:09": [
    {
      time: "5:09",
      book: "The Black Bag",
      author: "Louis Joseph Vance",
      prefix:
        "The primal flush of triumph which had saturated the American's humor on this signal success, proved but fictive and transitory when inquiry of the station attendants educed the information that the two earliest trains to be obtained were the ",
      suffix: " to Dunkerque and the 5:37 for Ostend.",
    },
  ],
  "05:10": [
    {
      time: "Ten minutes past five",
      book: "The Law and the Lady",
      author: "Wilkie Collins",
      prefix:
        '"Oh, my husband, I have done the deed which will relieve you of the wife whom you hate! I have taken the poison--all of it that was left in the paper packet, which was the first that I found. If this is not enough to kill me, I have more left in the bottle. ',
      suffix:
        ". \"You have just gone, after giving me my composing draught. My courage failed me at the sight of you. I thought to myself, 'If he look at me kindly, I will confess what I have done, and let him save my life.' You never looked at me at all. You only looked at the medicine. I let you go without saying a word.",
    },
    {
      time: "ten after five",
      book: "Bossypants",
      author: "Tina Fey",
      prefix:
        "I settled into a daily routine. Wake up at 4:40am, shower, get on the train north by ",
      suffix: ".",
    },
  ],
  "05:11": [
    {
      time: "eleven minutes past five",
      book: "The Hot Rock",
      author: "Donald E Westlake",
      prefix:
        "Today was Tuesday, the fifteenth of August; the sun had risen at ",
      suffix:
        " this morning and would set at two minutes before seven this evening.",
    },
  ],
  "05:12": [
    {
      time: "twelve minutes and six seconds past five o'clock",
      book: "Slummer's Paradise",
      author: "Herbert Asbury",
      prefix: "At ",
      suffix:
        " on the morning of April 18th, 1906, the San francisco peninsula began to shiver in the grip of an earthquake which, when its ultimate consequences are considered, was the most disastrous in the recorded history of the North American continent.",
    },
  ],
  "05:13": [
    {
      time: "5:13 am.",
      book: "Uptempo",
      author: "Nakia D Johnson",
      prefix:
        "Lying on my side in bed, I stared at my alarm clock until it became a blemish, its red hue glowing like a welcome sign beckoning me into the depths of hell's crimson-colored cavities. ",
      suffix:
        " To describe this Monday as a blue Monday was an understatement.",
    },
  ],
  "05:14": [
    {
      time: "5.14am",
      book: "Into the Web",
      author: "Thomas H Cook",
      prefix: "The time was ",
      suffix:
        ", a very strange time indeed for the sheriff to have seen what he claimed he saw as he made his early-morning rounds, first patrolling back and forth along the deserted, snowbound streets of Kingdom City before extending his vigilance northward, along County Road.",
    },
  ],
  "05:15": [
    {
      time: "5:15 a.m.",
      book: "Insomnia",
      author: "Stephen King",
      prefix: "By the first week of May, Ralph was waking up to birdsong at ",
      suffix:
        " He tried earplugs for a few nights, although he doubted from the outset that they would work. It wasn’t the newly returned birds that were waking him up, nor the occasional delivery-truck backfire out on Harris Avenue. He had always been the sort of guy who could sleep in the middle of a brass marching bad, and he didn’t think that had changed. What had changed was inside his head.",
    },
    {
      time: "5:15",
      book: "Fear and Loathing: On the Campaign Trail '72",
      author: "Hunter S Thompson",
      prefix:
        "Weird conversation with Brown, a tired & confused old man who's been jerked out of bed at ",
      suffix: ".",
    },
  ],
  "05:16": [
    {
      time: "5:16",
      book: "Fear and Loathing: On the Campaign Trail '72",
      author: "Hunter S Thompson",
      prefix: "",
      suffix:
        " - Mank on phone to Secretary of State Brown: 'Mr Brown, we're profoundly disturbed about this situation in the 21st. We can't get a single result out of there",
    },
    {
      time: "5:16 a.m",
      book: "The Accidental",
      author: "Ali Smith",
      prefix:
        "She could go back to sleep. But typical and ironic, she is completely awake. It is completely light outside now; you can see for miles. Except there is nothing to see here; trees and fields and that kind of thing. ",
      suffix: " on the substandard clock radio. She is really awake.",
    },
  ],
  "05:20": [
    {
      time: "five twenty",
      book: "The Periodic Table",
      author: "Primo Levi",
      prefix:
        "He saw on the floor his cigarette reduced to a long thin cylinder of ash: it had smoked itself. It was ",
      suffix:
        ", dawn was breaking behind the shed of empty barrels, the thermometer pointed to 210 degrees.",
    },
  ],
  "05:23": [
    {
      time: "5.23am",
      book: "The Tragedy of Arthur",
      author: "Arthur Phillips",
      prefix:
        "If I could count precisely to sixty between two passing orange minutes on her digital clock, starting at ",
      suffix:
        " and ending exactly as it melted into 5:24, then when she woke she would love me and not say this had been a terrible mistake.",
    },
  ],
  "05:24": [
    {
      time: "5:24",
      book: "The Tragedy of Arthur",
      author: "Arthur Phillips",
      prefix:
        "If I could count precisely to sixty between two passing orange minutes on her digital clock, starting at 523am. and ending exactly as it melted into ",
      suffix:
        ", then when she woke she would love me and not say this had been a terrible mistake.",
    },
  ],
  "05:25": [
    {
      time: "5.25",
      book: "Arthur and George",
      author: "Julian Barnes",
      prefix: "George's train home from New Street leaves at ",
      suffix: ". On the return journey, there are rarely schoolboys.",
    },
  ],
  "05:26": [
    {
      time: "05:26",
      book: "101 Reykjavik",
      author: "Hallgrímur Helgason",
      prefix:
        "I think this is actually bump number 1,970. And the boy keeps plugging away at the same speed. There isn’t a sound from them. Not a moan. Poor them. Poor me. I look at the clock. ",
      suffix: ".",
    },
  ],
  "05:28": [
    {
      time: "five-twenty-eight",
      book: "Hard Boiled Wonderland and the End of the World",
      author: "Haruki Murakami",
      prefix: "I pulled into the Aoyama supermarket parking garage at ",
      suffix:
        ". The sky to the east was getting light. I entered the store carrying my bag. Almost no one was in the place. A young clerk in a striped uniform sat reading a magazine; a woman of indeterminate age was buying a cartload of cans and instant food. I turned past the liquor display and went straight to the snack bar.",
    },
  ],
  "05:30": [
    {
      time: "half-past five",
      book: "An Insular Possession",
      author: "Timothy Mo",
      prefix:
        "Gideon has been most unlike Gideon. As Walter Eastman is preoccupied himself, he has not had time, or more to the point, inclination, to notice aberrant behaviour. For instance, it is ",
      suffix:
        " in the summer morning. Young Chase's narrow bachelor bed has evidently been slept in, for it is rumpled in that barely disturbed way which can never be counterfeited. His jug's empty and there's grey water in the basin, cleanly boy. The window is open, admitting the salubrious sea-breeze. He doesn't smoke anyway. What an innocent room it is.",
    },
    {
      time: "half-past five",
      book: "Jane Eyre",
      author: "Charlotte Brontë",
      prefix: "It was by this time ",
      suffix:
        ", and the sun was on the point of rising; but I found the kitchen still dark and silent. … The stillness of early morning slumbered everywhere .. the carriage horses stamped from time to time in their closed stables: all else was still.",
    },
    {
      time: "five-thirty",
      book: "Chronicle of a Death Foretold",
      author: "Gabriel García Márquez",
      prefix:
        "On the day they were going to kill him, Santiago Nasar got up at ",
      suffix: " in the morning to wait for the boat the bishop was coming on.",
    },
  ],
  "05:31": [
    {
      time: "5:31",
      book: "Fear and Loathing: On the Campaign Trail '72",
      author: "Hunter S Thompson",
      prefix: "",
      suffix:
        " - Mank on phone to lawyer: 'Jesus, I think we gotta go in there and get those ballots! Impound 'em! Every damn one!",
    },
  ],
  "05:34": [
    {
      time: "Five-thirty-four",
      book: "The Commodore",
      author: "C.S. Forester",
      prefix:
        "I asked \"What time is sunrise?”' A second's silence while the crestfallen Bush absorbed his rebuke, and then another voice answered: ‘",
      suffix: ", sir.'",
    },
  ],
  "05:35": [
    {
      time: "5:35",
      book: "Fear and Loathing: On the Campaign Trail '72",
      author: "Hunter S Thompson",
      prefix: "",
      suffix:
        " - All phones ringing now, the swing shift has shot the gap - now the others are waking up",
    },
    {
      time: "twenty-five before six",
      book: "the dice man",
      author: "Luke Rhinehart",
      prefix: "I squinted at the clock. 'It says ",
      suffix: ",' I said and rolled away from him.",
    },
  ],
  "05:37": [
    {
      time: "5:37",
      book: "This Book Will Save Your Life",
      author: "AM Homes",
      prefix: "Richard glanced at the clock on the microwave - ",
      suffix:
        " - almost twelve hours, almost one half-day since he'd dialed 911.",
    },
  ],
  "05:38": [
    {
      time: "5.38 a.m.",
      book: "Johnny Mackintosh: Battle for Earth",
      author: "Keith Mansfield",
      prefix:
        "Kovac,’ said Johnny sleepily. It was very rare for the quantum computer and not Sol to wake him up. ‘What’s going on? What time is it?’ ‘Good morning, Johnny,’ said the ship. ‘It is ",
      suffix:
        "’ ‘What?’ said Johnny. ‘It’s Saturday.’ ‘I told you he wouldn’t like it,’ said Sol, presumably to Kovac. ‘It’s hardly a matter of likes or dislikes,’ said the computer. ‘I have information I deem important enough to pass on at the earliest opportunity – whatever time it is.’",
    },
  ],
  "05:40": [
    {
      time: "Twenty minutes to six",
      book: "The Peculiar Memories of Thomas Penman",
      author: "Bruce Robinson",
      prefix: "",
      suffix:
        ". 'Rob's boys were already on the platform, barrows ready. The only thing that ever dared to be late around here was the train. Rob's boys were in fact Bill Bing, thirty, sucking a Woodbine, and Arthur, sixty, half dead",
    },
  ],
  "05:43": [
    {
      time: "5.43",
      book: "Fear and Loathing: On the Campaign Trail '72",
      author: "Hunter S. Thompson",
      prefix: "",
      suffix:
        " - Mank on phone to 'Mary' in Washington; 'It now appears quite clear that we'll lead the state - without the 21st.",
    },
  ],
  "05:45": [
    {
      time: "5:45",
      book: "IT",
      author: "Stephen King",
      prefix: "At ",
      suffix:
        " a power-transformer on a pole beside the abandoned Tracker Brothers’ Truck Depot exploded in a flash of purple light, spraying twisted chunks of metal onto the shingled roof.",
    },
  ],
  "05:46": [
    {
      time: "5.46am",
      book: "A Whispered Name",
      author: "William Brodrick",
      prefix:
        "Herbert could feel nothing. He wrote a legal-sounding phrase to the effect that the sentence had been carried out at ",
      suffix:
        ", adding, 'without a snag'. The burial party had cursed him quietly as they'd hacked at the thick roots and tight soil.",
    },
  ],
  "05:52": [
    {
      time: "5.52am",
      book: "Silent Witness",
      author: "Mark Fuhrman",
      prefix: "At ",
      suffix:
        " paramedics from the St. Petersburg Fire Department and SunStar Medic One ambulance service responded to a medical emergency call at 12201 Ninth Street North, St. Petersburg, apartment 2210.",
    },
  ],
  "05:55": [
    {
      time: "5.55am",
      book: "The Lost Luggage Porter",
      author: "Andrew Martin",
      prefix: "It was ",
      suffix:
        " and raining hard when I pedalled up to the bike stand just outside the forecourt of the station and dashed inside. I raced past the bookstall, where all the placards of the Yorkshire Post (a morning paper) read 'York Horror', but also 'Terrific February Gales at Coast'.",
    },
  ],
  "05:58": [
    {
      time: "5.58 a.m.",
      book: "The Girl who Kicked the Hornet's Nest",
      author: "Stieg Larsson",
      prefix: "Annika Giannini woke with a start. She saw that it was ",
      suffix: "",
    },
  ],
  "06:00": [
    {
      time: "six o’clock",
      book: "The Saints",
      author: "Patsy Hickman",
      prefix:
        "‘What’s the time?’ I ask, and telling him so that he knows, ‘My mother likes “peace and quiet” to sleep late on Saturday mornings.’ ‘She does, does she? It’s ",
      suffix:
        ". I couldn’t sleep,’ he says wearily, like an afterthought, as if it’s what he expects. ‘Why are you up so early?’ ‘I woke up and needed my panda. I can’t find him.’ ‘Where do you think he can be?’ His face changes and he smiles again, bending down to look under the table and behind the curtain. But he isn’t clowning or teasing. He’s in earnest.",
    },
    {
      time: "at six",
      book: "The Elegance of the Hedgehog",
      author: "Muriel Barbery",
      prefix:
        "But every morning, even if there's been a nighttime session and he has only slept two hours, he gets up ",
      suffix:
        " and reads his paper while he drinks a strong cup of coffee. In this way Papa constructs himself every day.",
    },
    {
      time: "at six a.m.",
      book: "Jane Eyre",
      author: "Charlotte Brontë",
      prefix:
        "I had risen half-an-hour before her entrance, and had washed my face, and put on my clothes by the light of a half-moon just setting, whose rays streamed through the narrow window near my crib. I was to leave Gateshead that day by a coach which passed the lodge gates ",
      suffix: "",
    },
    {
      time: "six",
      book: "Hunger",
      author: "Knut Hamsun",
      prefix: "Lying awake in my attic room, i hear a clock strike ",
      suffix:
        " downstairs. It was fairly light and people were beginning to walk up and down the stairs...- i heard the clock strike eight downstairs before i rose and got dressed... I looked up - the clock tower of our saviour's showed ten.",
    },
    {
      time: "six o'clock",
      book: "L'Education sentimentale",
      author: "Gustave Flaubert",
      prefix: "On the 15th of September 1840, about ",
      suffix:
        " in the morning, the Ville-de-Montereau, ready to depart, pouring out great whirls of smoke by the quai Saint-Bernard.",
    },
    {
      time: "6.00 A.M.",
      book: "The Great Gatsby",
      author: "F. Scott Fitzgerald",
      prefix: "Rise from bed ............... . ",
      suffix: "",
    },
    {
      time: "six",
      book: "The Leopard",
      author: "Giuseppe Tomasi di Lampedusa",
      prefix: "The ball went on for a long time, until ",
      suffix:
        " in the morning; all were exhausted and wishing they had been in bed for at least three hours; but to leave early was like proclaiming the party a failure and offending the host and hostess who had taken such a lot of trouble, poor dears.",
    },
  ],
  "06:02": [
    {
      time: "6.02",
      book: "Arthur and George",
      author: "Julian Barnes",
      prefix:
        "Bimingham New Street 5.25. Walsall 5.55. This train does not stop at Birchills, for reasons George has never been able to ascertain. Then it is Bloxwich ",
      suffix:
        ", Wyrley & Churchbridge 6.09. At 6.10 he nods to Mr Merriman the stationmaster.",
    },
  ],
  "06:05": [
    {
      time: "five minutes past six",
      book: "The ABC Murders",
      author: "Agatha Christie",
      prefix:
        "A second man went in and found the shop empty, as he thought, at ",
      suffix: ". That puts the time at between 5:30 and 6:05.",
    },
  ],
  "06:06": [
    {
      time: "6:06",
      book: "IT",
      author: "Stephen King",
      prefix: "At ",
      suffix:
        ", every toilet on Merit Street suddenly exploded in a geyser of shit and raw sewage as some unimaginable reversal took place in the pipes which fed the holding tanks of the new waste-treatment plant in the Barrens.",
    },
  ],
  "06:08": [
    {
      time: "six oh-eight a.m.",
      book: "Magic Bleeds",
      author: "Ilona Andrews",
      prefix: "At ",
      suffix:
        " two men wearing ragged trench coats approached the Casino. The shorter of the men burst into flames.",
    },
  ],
  "06:10": [
    {
      time: "ten past six",
      book: "The Member of the Wedding",
      author: "Carson McCullers",
      prefix: "The bus left the station at ",
      suffix:
        " - and she sat proud, like an accustomed traveller, apart from her father, John Henry, and Berenice. But after a while a serious doubt came in her, which even the answers of the bus-driver could not quite satisfy.",
    },
  ],
  "06:13": [
    {
      time: "06:13",
      book: "Room",
      author: "Emma Donoghue",
      prefix: "It's ",
      suffix:
        " .........Ma says I ought to be wrapped up in Rug already, Old Nick might possibly come.",
    },
  ],
  "06:15": [
    {
      time: "6.15",
      book: "The Great Gatsby",
      author: "F. Scott Fitzgerald",
      prefix: "Dumbbell exercise and wall-scaling ..... . ",
      suffix: "-6.30",
    },
    {
      time: "quarter past six",
      book: "A Clergyman's Daughter",
      author: "George Orwell",
      prefix: "Father expected his shaving-water to be ready at a ",
      suffix:
        ". Just seven minutes late, Dorothy took the can upstairs and knocked at her father's door.",
    },
    {
      time: "6.15 am.",
      book: "Girl Missing",
      author: "Sophie McKenzie",
      prefix: "It was ",
      suffix:
        " Just starting to get light. A small knot of older teenagers were leaning against a nearby wall. They looked as though they had been out all night.Two of the guys stared at us. Their eyes hard and threatening.",
    },
    {
      time: "6.15 am.",
      book: "Girl Missing",
      author: "Sophie McKenzie",
      prefix: "It was ",
      suffix:
        " Just starting to get light. A small knot of older teenagers were leaning against a nearby wall. They looked as though they had been out all night.Two of the guys stared at us. Their eyes hard and threatening.",
    },
  ],
  "06:17": [
    {
      time: "six-seventeen",
      book: "Lonely Hearts",
      author: "John Harvey",
      prefix:
        "Dizzy, come on.' He turned slowly, coaxing the animal down on to the pillow. The clock read ",
      suffix:
        ". A second cat, Miles, purred on contentedly from the patch in the covers where Resnick's legs had made a deep V.",
    },
  ],
  "06:19": [
    {
      time: "6.19 am",
      book: "Venus",
      author: "Carol Ann Duffy",
      prefix: "",
      suffix:
        ", 8th June 2004, the jet of your pupil set in the gold of your eye",
    },
  ],
  "06:20": [
    {
      time: "6:20 a.m.",
      book: "Soon I Will Be Invincible",
      author: "Austin Grossman",
      prefix: "It was ",
      suffix:
        ", and my parents and I were standing, stunned and haf-awake, in the parking lot of a Howard Johnson's in Iowa.",
    },
  ],
  "06:25": [
    {
      time: "6.25",
      book: "The Deaths",
      author: "Mark Lawson",
      prefix:
        "Simon is happy to travel scum class when he's on his own and even sometimes deliberately aims for the ",
      suffix: ". But today the .25 is delayed to 6.44.",
    },
    {
      time: "Six-twenty-five",
      book: "Hard Boiled Wonderland and the End of the World",
      author: "Haruki Murakami",
      prefix:
        "Still, it's your consciousness that's created it. Not somethin' just anyone could do. Others could be wanderin' around forever in who-knows-what contradictory chaos of a world. You're different. You seem t'be the immortal type.\" \"When's the turnover into that world going to take place?\" asked the chubby girl. The Professor looked at his watch. I looked at my watch. ",
      suffix:
        '. Well past daybreak. Morning papers delivered. "According t\'my estimates, in another twenty-nine hours and thirty-five minutes," said the Professor. "Plus or minus forty-five minutes. I set it at twelve noon for easy reference. Noon tomorrow.',
    },
  ],
  "06:27": [
    {
      time: "06:27",
      book: "Neuromancer",
      author: "William Gibson",
      prefix: "",
      suffix:
        ":52 by the chip in her optic nerve; Case had been following her progress through Villa Straylight for over an hour, letting the endorphin analogue she'd taken blot out his hangover",
    },
    {
      time: "0627 hours",
      book: "White Teeth",
      author: "Zadie Smith",
      prefix:
        "Early in the morning, late in the century, Cricklewood Broadway. At ",
      suffix:
        " on January 1, 1975, Alfred Archibald Jones was dressed in corduroy and sat in a fume-filled Cavalier Musketeer Estate, facedown on the steering wheel, hoping the judgment would not be too heavy upon him.",
    },
  ],
  "06:29": [
    {
      time: "a minute short of six-thirty",
      book: "The Big Sleep",
      author: "Raymond Chandler",
      prefix:
        "I sat up. There was a rug over me. I threw that off and got my feet on the floor. I scowled at a clock. The clock said ",
      suffix: ".",
    },
  ],
  "06:30": [
    {
      time: "6.30 am.",
      book: "Girl Missing",
      author: "Sophie McKenzie",
      prefix:
        "Inside now MJ ordered. She pushed the three of us into the hotel room, thern shut the soor. I glanced at the clock by the bed. ",
      suffix: " Why were they waking Mum and Dad up this early?",
    },
    {
      time: "six-thirty",
      book: "The Book of Daniel",
      author: "E.L. Doctorow",
      prefix:
        'Daniel and the FBI men listened to the sounds of his mother waking up his father. Daniel still held the door-knob. He was ready to close the door the second he was told to."What time is it?" said his father in a drugged voice. "Oh my God, it\'s ',
      suffix: '," his mother said.',
    },
    {
      time: "six-thirty",
      book: "The Man Who Loved Children",
      author: "Christina Stead",
      prefix: "It was ",
      suffix:
        ". When the baby's cry came, they could not pick it out, and Sam, eagerly thrusting his face amongst their ears, said, \"Listen, there, there, that's the new baby.\" He was red with delight and success.",
    },
    {
      time: "six-thirty",
      book: "Cities of the Plain",
      author: "Cormac McCarthy",
      prefix:
        "It was very cold sitting in the truck and after a while he got out and walked around and flailed at himself with his arms and stamped his boots. Then he got back in the truck. The bar clock said ",
      suffix:
        "...By eight-thirty he’d decided that it that was it would take to make the cab arrive then that’s what he would do and he started the engine.",
    },
    {
      time: "half-past six",
      book: "The Scarlet Pimpernel",
      author: "Baroness Orczy",
      prefix:
        "Nervously she jumped up and listened; the house itself was as still as ever; the footsteps had retreated. Through her wide-open window the brilliant rays of the morning sun were flooding her room with light. She looked up at the clock; it was ",
      suffix: "—too early for any of the household to be already astir.",
    },
    {
      time: "Six-thirty",
      book: "The Long Dark Tea-time of the Soul",
      author: "Douglas Adams",
      prefix: "",
      suffix:
        " was clearly a preposterous time and he, the client, obviously hadn't meant it seriously. A civilised six-thirty for twelve noon was almost certainly what he had in mind, and if he wanted to cut up rough about it, Dirk would have no option but to start handing out some serious statistics. Nobody got murdered before lunch. But nobody. People weren't up to it. You needed a good lunch to get both the blood-sugar and blood-lust levels up. Dirk had the figures to prove it",
    },
    {
      time: "6.30",
      book: "Period Piece",
      author: "Gwen Raverat",
      prefix:
        "Sometimes they were hooded carts, sometimes they were just open carts, with planks for seats, on which sat twelve cloaked and bonneted women, six a side, squeezed together, for the interminable journey. As late as 1914 I knew the carrier of Croydon-cum-Clopton, twelve miles from Cambridge; his cart started at ",
      suffix:
        " in the morning and got back at about ten at night. Though he was not old, he could neither read nor write; but he took commissions all along the road - a packet of needles for Mrs. This, and a new teapot for Mrs. That - and delivered them all correctly on the way back.",
    },
  ],
  "06:32": [
    {
      time: "twenty-eight minutes to seven",
      book: "Too Like the Lightning",
      author: "Dana Chambers",
      prefix:
        "The familiar radium numerals on my left wrist confirmed the clock tower. It was ",
      suffix:
        ". I seemed to be filling a set of loud maroon pajamas which were certainly not mine. My vis-a-vis was wearing a little number in yellow.",
    },
  ],
  "06:33": [
    {
      time: "6.33 a.m.",
      book: "The Voices of Time",
      author: "JG Ballard",
      prefix: "Woke ",
      suffix:
        " Last session with Anderson. He made it plain he's seen enough of me, and from now on I'm better alone. To sleep 8:00? (These count-downs terrify me.) He paused, then added: Goodbye, Eniwetok.",
    },
  ],
  "06:35": [
    {
      time: "twenty-five minutes to seven",
      book: "Ravensdene Court",
      author: "J.S. Fletcher",
      prefix:
        "My watch lay on the dressing-table close by; glancing at it, I saw that the time was ",
      suffix:
        ". I had been told that the family breakfasted at nine, so I had nearly two-and-a-half hours of leisure. Of course, I would go out, and enjoy the freshness of the morning.",
    },
  ],
  "06:36": [
    {
      time: "6:36",
      book: "The Voices of Time",
      author: "JG Ballard",
      prefix:
        "Kaldren pursues me like luminescent shadow. He has chalked up on the gateway '96,688,365,498,702'. Should confuse the mail man. Woke 9:05. To sleep ",
      suffix: ".",
    },
  ],
  "06:37": [
    {
      time: "6.37am",
      book: "American Gods",
      author: "Neil Gaiman",
      prefix: "The dashboard clock said ",
      suffix:
        " Town frowned, and checked his wristwatch, which blinked that it was 1.58pm. Great, he thought. I was either up on that tree for eight hours, or for minus a minute.",
    },
  ],
  "06:38": [
    {
      time: "6.38am.",
      book: "American Gods",
      author: "Neil Gaiman",
      prefix: "The clock on the dashboard said it was ",
      suffix: " He left the keys in the car, and walked toward the tree.",
    },
  ],
  "06:40": [
    {
      time: "twenty to seven",
      book: "The Long Dark Tea-Time of the Soul",
      author: "Douglas Adams",
      prefix:
        "At eleven o'clock the phone rang, and still the figure did not respond, any more than it has responded when the phone had rung at twenty-five to seven in the morning, and again at ",
      suffix: "",
    },
  ],
  "06:43": [
    {
      time: "6.43am.",
      book: "A View From the Foothills",
      author: "Chris Mullin",
      prefix: "To London on the ",
      suffix:
        " Jessica is back from her holiday. Things are looking up, she called me Chris, instead of Minister, when we talked on the phone this afternoon.",
    },
  ],
  "06:44": [
    {
      time: "6.44",
      book: "The Deaths",
      author: "Mark Lawson",
      prefix:
        "Simon is happy to travel scum class when he's on his own and even sometimes deliberately aims for the 6.25. But today the .25 is delayed to ",
      suffix: ".",
    },
  ],
  "06:45": [
    {
      time: "quarter to seven",
      book: "No Name",
      author: "Wilkie Collins",
      prefix: "As the clock pointed to a ",
      suffix:
        ", the dog woke and shook himself. After waiting in vain for the footman, who was accustomed to let him out, the animal wandered restlessly from one closed door to another on the ground floor; and, returning to his mat in great perplexity, appealed to the sleeping family, with a long and melancholy howl.'",
    },
    {
      time: "quarter to seven",
      book: "Metamorphosis",
      author: "Franz Kafka",
      prefix:
        "He was still hurriedly thinking all this through, unable to decide to get out of the bed, when the clock struck ",
      suffix:
        '. There was a cautious knock at the door near his head. "Gregor", somebody called - it was his mother - "it\'s quarter to seven. Didn\'t you want to go somewhere?"',
    },
  ],
  "06:46": [
    {
      time: "one minute after the quarter to seven",
      book: "The Thirty-Nine Steps",
      author: "John Buchan",
      prefix: "At ",
      suffix:
        " I heard the rattle of the cans outside. I opened the front door, and there was my man, singling out my cans from a bunch he carried and whistling through his teeth.",
    },
    {
      time: "one minute after the quarter to seven",
      book: "The Thirty-Nine Steps",
      author: "John Buchan",
      prefix:
        "Then I hung about in the hall waiting for the milkman. That was the worst part of the business, for I was fairly choking to get out of doors. Six-thirty passed, then six-forty, but still he did not come. The fool had chosen this day of all days to be late. At ",
      suffix:
        " I heard the rattle of the cans outside. I opened the front door, and there was my man, singling out my cans from a bunch he carried and whistling through his teeth. He jumped a bit at the sight of me.",
    },
  ],
  "06:49": [
    {
      time: "6:49",
      book: "Fear and Loathing: On the Campaign Trail '72",
      author: "Hunter S. Thompson",
      prefix: "Night ends, ",
      suffix: ". Meet in the coffee shop at 7:30; press conference at 10:00.",
    },
  ],
  "06:50": [
    {
      time: "six-fifty",
      book: "Pretty Ice",
      author: "Mary Robison",
      prefix: "Will, my fiancé, was coming from Boston on the ",
      suffix:
        " train - the dawn train, the only train that still stopped in the small Ohio city where I lived.",
    },
  ],
  "06:55": [
    {
      time: "6:55 am",
      book: "What was Lost",
      author: "Catherine O'Flynn",
      prefix: "At ",
      suffix:
        " Lisa parked and took the lift from the frozen underground car park up to level 1 of Green Oaks Shopping Centre.",
    },
  ],
  "06:59": [
    {
      time: "6.59 a.m.",
      book: "The Girl who Played with Fire",
      author: "Stieg Larsson",
      prefix: "It was ",
      suffix:
        ' on Maundy Thursday as Blomkvist and Berger let themselves into the "Millennium" offices.',
    },
  ],
  "07:00": [
    {
      time: "Seven o'clock",
      book: "Metamorphosis",
      author: "Franz Kafka",
      prefix: '"',
      suffix:
        ', already", he said to himself when the clock struck again, "seven o\'clock, and there\'s still a fog like this."',
    },
    {
      time: "seven o’clock",
      book: "Darkness at Noon",
      author: "Arthur Koestler",
      prefix: "At ",
      suffix:
        " in the morning, Rubashov was awakened by a bugle, but he did not get up. Soon he heard sounds in the corridor. He imagined that someone was to be tortured, and he dreaded hearing the first screams of pain. When the footsteps reached his own section, he saw through the eye hole that guards were serving breakfast. Rubashov did not receive any breakfast because he had reported himself ill. He began to pace up and down the cell, six and a half steps to the window, six and a half steps back.",
    },
    {
      time: "at seven",
      book: "Great Expectations",
      author: "Charles Dickens",
      prefix: "I had left directions that I was to be called ",
      suffix:
        "; for it was plain that I must see Wemmick before seeing any one else, and equally plain that this was a case in which his Walworth sentiments, only, could be taken. It was a relief to get out of the room where the night had been so miserable, and I needed no second knocking at the door to startle me from my uneasy bed.",
    },
    {
      time: "seven o'clock",
      book: "Crime and Punishment",
      author: "Fyodor Dostoyevsky",
      prefix:
        "She locked herself in, made no reply to my bonjour through the door; she was up at ",
      suffix: ", the samovar was taken in to her from the kitchen.",
    },
  ],
  "07:02": [
    {
      time: "07:02",
      book: "Neuromancer",
      author: "William Gibson",
      prefix: "",
      suffix: ":18 One and a half hours. 'Case,' she said, 'I wanna favour.",
    },
  ],
  "07:03": [
    {
      time: "7:03am",
      book: "The Night of the Generals",
      author: "Hans Hellmut Kirst",
      prefix: "",
      suffix: " General Tanz woke up as though aroused by a mental alarm-clock",
    },
  ],
  "07:04": [
    {
      time: "7:04 p.m.",
      book: "The Lost Honour of Katharina Blum",
      author: "Heinrich Böll",
      prefix:
        "Sunday evening at almost the same hour (to be precise, at about ",
      suffix:
        ") she rings the front door bell at the home of Walter Moeding, Crime Commissioner, who is at that moment engaged, for professional rather than private reasons, in disguising himself as a sheikh.",
    },
  ],
  "07:05": [
    {
      time: "five minutes after seven o'clock",
      book: "Where I'm Calling From",
      author: "Raymond Carver",
      prefix:
        "He really couldn't believe that the old woman who'd phoned him last night would show up this morning, as she'd said she would. He decided he'd wait until ",
      suffix:
        ", and then he'd call in, take the day off, and make every effort in the book to locate someone reliable.",
    },
    {
      time: "five after seven",
      book: "Dance Dance Dance",
      author: "Haruki Murakami",
      prefix:
        "Outside my window the sky hung low and gray. It looked like snow, which added to my malaise. The clock read ",
      suffix:
        ". I punched the remote control and watched the morning news as I lay in bed.",
    },
    {
      time: "7:05 A.M.",
      book: "The Hunt for Red October",
      author: "Tom Clancy",
      prefix:
        "Ryan missed the dawn. He boarded a TWA 747 that left Dulles on time, at ",
      suffix:
        " The sky was overcast, and when the aircraft burst through the cloud layer into sunlight, Ryan did something he had never done before. For the first time in his life, Jack Ryan fell asleep on an airplane.",
    },
  ],
  "07:06": [
    {
      time: "six minutes past seven",
      book: "The Green Mile",
      author: "Stephen King",
      prefix:
        "So far so good. There followed a little passage of time when we stood by the duty desk, drinking coffee and studiously not mentioning what we were all thinking and hoping: that Percy was late, that maybe Percy wasn't going to show up at all. Considering the hostile reviews he'd gotten on the way he'd handled the electrocution, that seemed at least possible. But Percy subscribed to that old axiom about how you should get right back on the horse that had thrown you, because here he came through the door at ",
      suffix:
        ", resplendent in his blue uniform with his sidearm on one hip and his hickory stick in its ridiculous custom-made holster on the other.",
    },
    {
      time: "six minutes past seven",
      book: "The Green Mile",
      author: "Stephen King",
      prefix:
        "Percy subscribed to that old axiom about how you should get right back on the horse that had thrown you, because here he came through the door at ",
      suffix:
        ", resplendent in his blue uniform with his sidearm on one hip and his hickory stick in its ridiculous custom-made holster on the other.",
    },
  ],
  "07:08": [
    {
      time: "between eight and nine minutes after seven o'clock",
      book: "The Hard Way",
      author: "Lee Child",
      prefix:
        "Reacher had no watch but he figured when he saw Gregory it must have been ",
      suffix: ".",
    },
  ],
  "07:09": [
    {
      time: "Seven-nine",
      book: "There Will Come Soft Rains",
      author: "Ray Bradbury",
      prefix:
        "In the living room the voice-clock sang, Tick-tock, seven o'clock, time to get up, time to get up, seven o 'clock! as if it were afraid that nobody would. The morning house lay empty. The clock ticked on, repeating and repeating its sounds into the emptiness. ",
      suffix: ", breakfast time, seven-nine!",
    },
    {
      time: "Seven-nine",
      book: "There Will Come Soft Rains",
      author: "Ray Bradbury",
      prefix: "",
      suffix: ", breakfast time, seven-nine",
    },
  ],
  "07:10": [
    {
      time: "7.10",
      book: "The Thirty-Nine Steps",
      author: "John Buchan",
      prefix:
        "A search in Bradshaw informed me that a train left St Pancras at ",
      suffix:
        ", which would land me at any Galloway station in the late afternoon.",
    },
    {
      time: "7:10",
      book: "The Fourth Passenger",
      author: "Mini Nair",
      prefix:
        "There were many others waiting to execute the same operation, so she would have to move fast, elbow her way to the front so that she emerged first. The time was ",
      suffix:
        " in the morning. The manoeuvre would start at 7:12. She looked apprehensively at the giant clock at the railway station.",
    },
  ],
  "07:12": [
    {
      time: "7:12",
      book: "A Fraction of the Whole",
      author: "Steve Toltz",
      prefix:
        "He taught me that if I had to meet someone for an appointment, I must refuse to follow the 'stupid human habit' of arbitrarily choosing a time based on fifteen-minute intervals. 'Never meet people at 7:45 or 6:30, Jasper, but pick times like ",
      suffix: " and 8:03!'",
    },
  ],
  "07:13": [
    {
      time: "seven-thirteen",
      book: "Austerlitz",
      author: "WG Sebald",
      prefix:
        "It was all the more surprising and indeed alarming a little later, said Austerlitz, when I looked out of the corridor window of my carriage just before the train left at ",
      suffix:
        ", to find it dawning upon me with perfect certainty that I had seen the pattern of glass and steel roof above the platforms before.",
    },
  ],
  "07:14": [
    {
      time: "7.14",
      book: "The Redeemer",
      author: "Jo Nesbo",
      prefix: "At ",
      suffix:
        " Harry knew he was alive. He knew that because the pain could be felt in every nerve fibre.",
    },
  ],
  "07:15": [
    {
      time: "7:15 A.M.",
      book: "At the Mountains of Madness",
      author: "H.P. Lovecraft",
      prefix: "At ",
      suffix:
        ", January 25th, we started flying northwestward under McTighe's pilotage with ten men, seven dogs, a sledge, a fuel and food supply, and other items including the plane's wireless outfit.",
    },
    {
      time: "7.15",
      book: "The Suspicions of Mr Whicher",
      author: "Kate Summerscale",
      prefix:
        "Gough again knocked on Mr and Mrs Kent's bedroom door. This time it was opened - Mary Kent had got out of bed and put on her dressing gown, having just checked her husband's watch: it was ",
      suffix:
        ". A confused conversation ensued, in which each woman seemed to assume Saville was with the other.",
    },
    {
      time: "7.15",
      book: "The Suspicions of Mr Whicher",
      author: "Kate Summerscale",
      prefix:
        "Gough again knocked on Mr and Mrs Kent's bedroom door. This time it was opened - Mary Kent had got out of bed and put on her dressing gown, having just checked her husband's watch: it was ",
      suffix:
        ". A confused conversation ensued, in which each woman seemed to assume Saville was with the other.",
    },
    {
      time: "quarter-past seven",
      book: "The Adventure of the Speckled Band",
      author: "Arthur Conan Doyle",
      prefix:
        "It was early in April in the year ’83 that I woke one morning to find Sherlock Holmes standing, fully dressed, by the side of my bed. He was a late riser, as a rule, and as the clock on the mantelpiece showed me that it was only a ",
      suffix:
        ", I blinked up at him in some surprise, and perhaps just a little resentment, for I was myself regular in my habits.",
    },
  ],
  "07:17": [
    {
      time: "7.17am",
      book: "Against the Day",
      author: "Thomas Pynchon",
      prefix: "As of ",
      suffix:
        " local time on 30 June 1908, Padzhitnoff had been working for nearly a year as a contract employee of the Okhrana, receiving five hundred rubles a month, a sum which hovered at the exorbitant end of spy-budget outlays for those years.",
    },
  ],
  "07:19": [
    {
      time: "7.19am",
      book: "The Rosie Project",
      author: "Graeme Simsion",
      prefix:
        "I opened the sunroof and turned up the CD player volume to combat fatigue, and at ",
      suffix:
        " on Saturday, with the caffeine still running all around my brain, Jackson Browne and I pulled into Moree.",
    },
  ],
  "07:20": [
    {
      time: "7.20 a.m.",
      book: "The Curious Incident of the Dog in the Night-time",
      author: "Mark Haddon",
      prefix:
        "And this was my timetable when I lived at home with Father and I thought that Mother was dead from a heart attack (this was the timetable for a Monday and also it is an approximation). ",
      suffix: " Wake up",
    },
    {
      time: "seven-twenty",
      book: "Babbitt",
      author: "Sinclair Lewis",
      prefix:
        "He who had been a boy very credulous of life was no longer greatly interested in the possible and improbable adventures of each new day. He escaped from reality till the alarm-clock rang, at ",
      suffix: ".",
    },
  ],
  "07:25": [
    {
      time: "7.25 a.m.",
      book: "The Curious Incident Of The Dog In The Night-Time",
      author: "Mark Haddon",
      prefix: "",
      suffix: " clean teeth and wash fac",
    },
  ],
  "07:27": [
    {
      time: "7.27",
      book: "The Return of the Dancing Master",
      author: "Henning Mankell",
      prefix: "His appointment with the doctor was for 8.45. It was ",
      suffix: ".",
    },
  ],
  "07:29": [
    {
      time: "7.29",
      book: "At Break of Day",
      author: "Elizabeth Speller",
      prefix: "At ",
      suffix:
        " in the morning of 1 July, the cinematographer finds himself filming silence itself.",
    },
  ],
  "07:30": [
    {
      time: "half-past seven",
      book: "After Rain",
      author: "William Trevor",
      prefix: "At ",
      suffix: " the next morning he rang the bell of 21 Blenheim Avenue.",
    },
    {
      time: "half past seven",
      book: "Closely Observed Trains",
      author: "Bohumil Hrabal",
      prefix: "Precisely at ",
      suffix:
        " the station-master came into the traffic office. He weighed almost sixteen stone, but women always said that he was incredibly light on his feet when he danced.",
    },
  ],
  "07:32": [
    {
      time: "7:32",
      book: "IT",
      author: "Stephen King",
      prefix: "At ",
      suffix: ", he suffered a fatal stroke.",
    },
  ],
  "07:34": [
    {
      time: "7:34.",
      book: "Let The Right One In",
      author: "John Ajvide Lindqvist",
      prefix: "",
      suffix:
        " Monday morning, Blackeberg. The burglar alarm at the ICA grocery store on Arvid Morne's way is set off",
    },
  ],
  "07:35": [
    {
      time: "7:35am",
      book: "The Devotion of Duspect X",
      author: "Higashino, Keigo",
      prefix: "At ",
      suffix: " Ishigami left his apartment as he did every weekday morning.",
    },
    {
      time: "Seven thirty-five",
      book: "Bare Bones",
      author: "Kathy Reichs",
      prefix: "I looked at my watch. ",
      suffix: ".",
    },
  ],
  "07:36": [
    {
      time: "7:36",
      book: "Let The Right One In",
      author: "John Ajvide Lindqvist",
      prefix: "",
      suffix:
        ", sunrise. The hospital blinds were much better, darker than her own",
    },
  ],
  "07:39": [
    {
      time: "7.39",
      book: "Arthur & George",
      author: "Julian Barnes",
      prefix:
        "Now, at the station, do you recall speaking to Mr Joseph Markew?' 'Yes, indeed. I was standing on the platform waiting for my usual train - the ",
      suffix: " - when he accosted me.'",
    },
  ],
  "07:40": [
    {
      time: "7.40 a.m.",
      book: "The Curious Incident Of The Dog In The Night-Time",
      author: "Mark Haddon",
      prefix: "",
      suffix: " Have breakfast",
    },
  ],
  "07:42": [
    {
      time: "Seven forty-two",
      book: "Magic Bleeds",
      author: "Ilona Andrews",
      prefix: "",
      suffix:
        " am., Mr Gasparian: I curse you. I curse your arms so they will wither and die and fall off your body..",
    },
  ],
  "07:44": [
    {
      time: "seven forty-four",
      book: "One moment, one morning",
      author: "Sarah Rayner",
      prefix:
        "And there I was, complaining that all this was just inconvenient, Anna castigates herself. The Goth was obviously right. What does it matter, really, if I'm a bit late for work? She voices her thoughts: \"It's not exactly how you'd choose to go, is it? You'd rather die flying a kite with your grandchildren, or at a great party or something. Not on the ",
      suffix: '."',
    },
    {
      time: "seven forty-four",
      book: "One Moment, One Morning",
      author: "Sarah Rayner",
      prefix:
        "The Goth was obviously right. What does it matter, really, if I'm a bit late for work? She voices her thoughts: \"It's not exactly how you'd choose to go, is it? You'd rather die flying a kite with your grandchildren, or at a great party or something. Not on the ",
      suffix: '."',
    },
  ],
  "07:45": [
    {
      time: "quarter to eight",
      book: "A Crime in The Neighborhood",
      author: "Suzanne Berne",
      prefix: "Mr Green left for work at a ",
      suffix:
        ", as he did every morning. He walked down his front steps carrying his empty-looking leatherette briefcase with the noisy silver clasps, opened his car door, and ducked his head to climb into the driver's seat.",
    },
    {
      time: "quarter to eight",
      book: "A crime in the neighborhood",
      author: "Suzanne Berne",
      prefix: "Mr Green left for work at a ",
      suffix:
        ", as he did every morning. He walked down his front steps carrying his empty-looking leatherette briefcase with the noisy silver clasps, opened his car door, and ducked his head to climb into the driver's seat.",
    },
  ],
  "07:46": [
    {
      time: "7.46 a.m.",
      book: "The Dogs of Riga",
      author: "Henning Mankell",
      prefix: "He awoke with a start. The clock on his bedside table said ",
      suffix:
        " He cursed, jumped out of bed and dressed. He stuffed his toothbrush and toothpaste in his jacket pocket, and parked outside the station just before 8 a.m. In reception, Ebba beckoned to him.",
    },
  ],
  "07:50": [
    {
      time: "ten minutes to eight",
      book: "The Ragged Trousered Philanthropists",
      author: "Robert Tressell",
      prefix: "At about ",
      suffix:
        ", Jim had squared the part of the work he had been doing - the window - so he decided not to start on the door or the skirting until after breakfast.",
    },
  ],
  "07:51": [
    {
      time: "nine minutes to eight",
      book: "Thud!",
      author: "Terry Pratchett",
      prefix:
        'Vimes fished out the Gooseberry as a red-hot cabbage smacked into the road behind him. "Good morning!" he said brightly to the surprised imp. "What is the time, please?" "Er...',
      suffix: ', Insert Name Here," said the imp.',
    },
  ],
  "07:53": [
    {
      time: "Seven to eight",
      book: "Never go back",
      author: "Robert Goddard",
      prefix: '"What time is it?" "',
      suffix: ". Won't be long now ...\"",
    },
  ],
  "07:55": [
    {
      time: "7.55",
      book: "Tightrope, from Selected Poems 1967-1987",
      author: "Roger McGough",
      prefix: "at ",
      suffix: " this morning the circus ran away to join me.",
    },
  ],
  "07:56": [
    {
      time: "seven fifty-six",
      book: "The Undomestic Goddess",
      author: "Sophie Kinsella",
      prefix:
        "I sit by the window, crunching toast, sipping coffee, and leafing through the paper in a leisurely way. At last, after devouring three slices, two cups of coffee, and all the Saturday sections, I stretch my arms in a big yawn and glance at the clock. I don't believe it. It's only ",
      suffix: ".",
    },
    {
      time: "four minutes to eight",
      book: "Buddenbrooks",
      author: "Thomas Mann",
      prefix: "The Castle Gate - only the Castle Gate - and it was ",
      suffix: ".",
    },
  ],
  "07:59": [
    {
      time: "7.59",
      book: "11/22/63",
      author: "Stephen King",
      prefix: "I'd spent fifty two days in 1958, but here it was ",
      suffix: " in the morning.",
    },
  ],
  "08:00": [
    {
      time: "8 a.m.",
      book: "Play it as is Lays",
      author: "Joan Didion",
      prefix:
        '"I\'m not crying," Maria said when Carter called from the desert at ',
      suffix: ' "I\'m perfectly alright". "You don\'t sound perfectly alright',
    },
    {
      time: "8.00 a.m.",
      book: "The Curious Incident of the Dog in the Night-time",
      author: "Mark Haddon",
      prefix: "",
      suffix: " Put school clothes o",
    },
    {
      time: "8 o'clock",
      book: "Hitch-Hikers Guide to the Galaxy",
      author: "Douglas Adams",
      prefix: "At ",
      suffix: " on Thursday morning Arthur didn't feel very good.",
    },
    {
      time: "eight o'clock",
      book: "Hitch-hikers guide to the galaxy",
      author: "Douglas Adams",
      prefix: "At ",
      suffix:
        " on Thursday morning Arthur didn't feel very good. He woke up blearily, got up, wandered blearily round his room, opened a window, saw a bulldozer, found his slippers and stomped off to the bathroom to wash.",
    },
    {
      time: "At eight o’clock",
      book: "Journey to the Centre of the Earth",
      author: "Jules Verne",
      prefix: "",
      suffix:
        ", a shaft of daylight came to wake us. The thousand facets of the lava on the rock face picked it up as it passed, scattering like a shower of sparks",
    },
    {
      time: "eight o'clock",
      book: "Brooklyn Follies",
      author: "Paul Auster",
      prefix: "But for now it was still ",
      suffix:
        ", and as I walked along the avenue under that brilliant blue sky, I was happy, my friends, as happy as any man who had ever lived.",
    },
    {
      time: "eight o'clock",
      book: "City of Glass",
      author: "Paul Auster",
      prefix: "By ",
      suffix:
        " Stillman would come out, always in his long brown overcoat, carrying a large, old-fashioned carpet bag. For two weeks this routine did not vary. The old man would wander through the streets of the neighbourhood, advancing slowly, sometimes by the merest of increments, pausing, moving on again, pausing once more, as though each step had to be weighed and measured before it could take its place among the sum total of steps.",
    },
    {
      time: "at eight",
      book: "Solar",
      author: "Ian McEwan",
      prefix:
        "Dressed in sweater, anorak and long johns, he lay in bed, hemmed in on three sides by chunky wooden beams, and ate all the salted snacks in the minibar, and then all the sugary snacks, and when he was woken by reception ",
      suffix:
        " the following morning to be told that everyone was waiting for him downstairs, the wrapper of a Mars bar was still folded in his fist.",
    },
    {
      time: "at eight",
      book: "One Flew Over the Cuckoo's Nest",
      author: "Ken Kesey",
      prefix:
        "I hear noise at the ward door, off up the hall out of my sight. That ward door starts opening ",
      suffix: " and opens and closes a thousand times a day, kashash, click.",
    },
    {
      time: "eight o'clock",
      book: "Pride and Prejudice",
      author: "Jane Austen",
      prefix: "It was dated from Rosings, at ",
      suffix:
        ' in the morning, and was as follows: - "Be not alarmed, madam, on receiving this letter, by the apprehension of its containing any repetition of those sentiments or renewal of those offerings which were last night so disgusting to you.',
    },
    {
      time: "eight o'clock",
      book: "Great Expectations",
      author: "Charles Dickens",
      prefix: "Mr. Pumblechook and I breakfasted at ",
      suffix:
        " in the parlour behind the shop, while the shopman took his mug of tea and hunch of bread-and-butter on a sack of peas in the front premises.",
    },
    {
      time: "eight o'clock a.m.",
      book: "Jane Eyre",
      author: "Charlotte Brontë",
      prefix:
        "Mrs. Rochester! She did not exist: she would not be born till to-morrow, some time after ",
      suffix:
        "; and I would wait to be assured she had come into the world alive, before I assigned to her all that property.",
    },
    {
      time: "eight",
      book: "A shropshire Lad",
      author: "A E Housman",
      prefix:
        "So here I'll watch the night and wait To see the morning shine, When he will hear the stroke of ",
      suffix: " And not the stroke of nine;",
    },
    {
      time: "eight o'clock",
      book: "The Trial",
      author: "Franz Kafka",
      prefix:
        "Someone must have been telling lies about Joseph K. for without having done anything wrong he was arrested one fine morning. His landlady's cook, who always brought him breakfast at ",
      suffix: ", failed to appear on this occasion.",
    },
    {
      time: "oh eight oh oh hours",
      book: "A Clockwork Orange",
      author: "Anthony Burgess",
      prefix: "The next morning I woke up at ",
      suffix:
        ", my brothers, and as I still felt shagged and fagged and fashed and bashed and as my glazzies were stuck together real horrorshow with sleepglue, I thought I would not go to school .",
    },
    {
      time: "eight o'clock",
      book: "Anna Karenina",
      author: "Leo Tolstoy",
      prefix:
        "Three days after the quarrel, Prince Stepan Arkadyevitch Oblonsky--Stiva, as he was called in the fashionable world-- woke up at his usual hour, that is, at ",
      suffix:
        " in the morning, not in his wife's bedroom, but on the leather-covered sofa in his study.",
    },
    {
      time: "exactly eight",
      book: "Three Men and a Maid",
      author: "P.G. Wodehouse",
      prefix:
        "Through the curtained windows of the furnished apartment which Mrs. Horace Hignett had rented for her stay in New York rays of golden sunlight peeped in like the foremost spies of some advancing army. It was a fine summer morning. The hands of the Dutch clock in the hall pointed to thirteen minutes past nine; those of the ormolu clock in the sitting-room to eleven minutes past ten; those of the carriage clock on the bookshelf to fourteen minutes to six. In other words, it was ",
      suffix:
        "; and Mrs. Hignett acknowledged the fact by moving her head on the pillow, opening her eyes, and sitting up in bed. She always woke at eight precisely.",
    },
    {
      time: "at eight",
      book: "Death in Venice",
      author: "Thomas Mann",
      prefix:
        "When he opened the windows in the morning, the sky was as overcast as it had been, but the air seemed fresher, and regret set in. Had giving notice not been impetuous and wrongheaded, the result of an inconsequential indisposition? If he had held off a bit, if he had not been so quick to lose heart, if he had instead tried to adjust to the air or wait for the weather to improve, he would now have been free of stress and strain and looking forward to a morning on the beach like the one the day before. Too late. He must go on wanting what he had wanted yesterday. He dressed and rode down to the ground floor ",
      suffix: " for breakfast.",
    },
  ],
  "08:01": [
    {
      time: "Eight-one",
      book: "There Will Come Soft Rains",
      author: "Ray Bradbury",
      prefix: "",
      suffix:
        ", tick-tock, eight-one o'clock, off to school, off to work, run, run, eight-one",
    },
  ],
  "08:02": [
    {
      time: "Eight oh two",
      book: "Jingo",
      author: "Terry Pratchett",
      prefix: "... bingeley ... ",
      suffix:
        " eh em, Death of Corporal Littlebottombottom ... Eight oh three eh em ... Death of Sergeant Detritus ... Eight oh threethreethree eh em and seven seconds seconds ... Death of Constable Visit ... Eight oh three eh em and nineninenine seconds ... Death of death of death of ...",
    },
  ],
  "08:03": [
    {
      time: "Eight oh three",
      book: "Jingo",
      author: "Terry Pratchett",
      prefix:
        "... bingeley ... Eight oh two eh em, Death of Corporal Littlebottombottom ... ",
      suffix:
        " eh em ... Death of Sergeant Detritus ... Eight oh threethreethree eh em and seven seconds seconds ... Death of Constable Visit ... Eight oh three eh em and nineninenine seconds ... Death of death of death of ...",
    },
    {
      time: "8:03",
      book: "A Fraction of the Whole",
      author: "Steve Toltz",
      prefix:
        "He taught me that if I had to meet someone for an appointment, I must refuse to follow the 'stupid human habit' of arbitrarily choosing a time based on fifteen-minute intervals. 'Never meet people at 7:45 or 6:30, Jasper, but pick times like 7:12 and ",
      suffix: "!'",
    },
  ],
  "08:04": [
    {
      time: "8:04",
      book: "The Periodic Table",
      author: "Primo Levi",
      prefix:
        "... every clerk had his particular schedule of hours, which coincided with a single pair of tram runs coming from the city: A had to come in at 8, B at ",
      suffix:
        ", C at 8:08 and so on, and the same for quitting times, in such a manner that never would two colleagues have the opportunity to travel in the same tramcar.",
    },
  ],
  "08:05": [
    {
      time: "8.05 a.m.",
      book: "The Curious Incident Of The Dog In The Night-Time",
      author: "Mark Haddon",
      prefix: "",
      suffix: " Pack school ba",
    },
  ],
  "08:08": [
    {
      time: "8:08",
      book: "The Periodic Table",
      author: "Primo Levi",
      prefix:
        "... every clerk had his particular schedule of hours, which coincided with a single pair of tram runs coming from the city: A had to come in at 8, B at 8:04, C at ",
      suffix:
        " and so on, and the same for quitting times, in such a manner that never would two colleagues have the opportunity to travel in the same tramcar.",
    },
  ],
  "08:09": [
    {
      time: "8:09",
      book: "American Tabloid",
      author: "James Ellroy",
      prefix: "He followed the squeals down a hallway. A wall clock read ",
      suffix: " - 10:09 Dallas time.",
    },
  ],
  "08:10": [
    {
      time: "8.10 a.m.",
      book: "The Curious Incident of the Dog in the Night-time",
      author: "Mark Haddon",
      prefix: "",
      suffix: " Read book or watch vide",
    },
    {
      time: "8:10",
      book: "This Side of Paradise",
      author: "F. Scott Fitzgerald",
      prefix:
        "Amory rushed into the house and the rest followed with a limp mass that they laid on the sofa in the shoddy little front parlor. Sloane, with his shoulder punctured, was on another lounge. He was half delirious, and kept calling something about a chemistry lecture at ",
      suffix: ".",
    },
    {
      time: "8:10",
      book: "The Voices of Time",
      author: "JG Ballard",
      prefix: "Cell count down to 400,000. Woke ",
      suffix:
        ". To sleep 7:15. (Appear to have lost my watch without realising it, had to drive into town to buy another.)",
    },
  ],
  "08:11": [
    {
      time: "eight-eleven",
      book: "The Blackpool Highflyer",
      author: "Andrew Martin",
      prefix:
        "'Care for a turn on the engine?' he called to the doxies, and pointed up at the footplate. They laughed but voted not to, climbing up with their bathtub into one of the rattlers instead. They both had very fetching hats, with one flower apiece, but the prettiness of their faces made you think it was more. For some reason they both wore white rosettes pinned to their dresses. I looked again at the clock: ",
      suffix: ".",
    },
  ],
  "08:12": [
    {
      time: "8:12 a.m.",
      book: "In Time Which Made A Monkey Of Us All",
      author: "Grace Paley",
      prefix: "At ",
      suffix:
        ", just before the moment of pff, all the business of the cellars was being transacted - garbage transferred from small cans into large ones; early wide-awake grandmas, rocky with insomnia, dumped wash into the big tubs; boys in swimming trunks rolled baby carriages out into the cool morning.",
    },
  ],
  "08:13": [
    {
      time: "8:13 a.m.",
      book: "In Time Which Made A Monkey Of Us All",
      author: "Grace Paley",
      prefix: "At ",
      suffix:
        " the alarm clock in the laboratory gave the ringing word. Eddie touched a button in the substructure of an ordinary glass coffeepot, from whose spout two tubes proceeded into the wall.",
    },
  ],
  "08:15": [
    {
      time: "quarter-past eight",
      book: "Three Men in a Boat",
      author: "Jerome K Jerome",
      prefix:
        "It was in the winter when this happened, very near the shortest day, and a week of fog into the bargain, so the fact that it was still very dark when George woke in the morning was no guide to him as to the time. He reached up, and hauled down his watch. It was a ",
      suffix: ".",
    },
    {
      time: "eight fifteen",
      book: "Pale Fire",
      author: "Vladimir Nabokov",
      prefix: "You scrutinized your wrist: \"It's ",
      suffix:
        ". (And here time forked.) I'll turn it on.\" The screen In its blank broth evolved a lifelike blur, And music welled.",
    },
  ],
  "08:16": [
    {
      time: "eight sixteen",
      book: "The Undomestic Goddess",
      author: "Sophie Kinsella",
      prefix:
        "I walk through the fruit trees toward a huge, square, brown patch of earth with vegetation growing in serried rows. These must be the vegetables. I prod one of them cautiously with my foot. It could be a cabbage or a lettuce. Or the leaves of something growing underground, maybe. To be honest, it could be an alien. I have no idea. I sit down on a mossy wooden bench and look at a nearby bush covered in white flowers. Mm. Pretty. Now what? What do people do in their gardens? I feel I should have something to read. Or someone to call. My fingers are itching to move. I look at my watch. Still only ",
      suffix: ". Oh God.",
    },
  ],
  "08:17": [
    {
      time: "8.17 a.m.",
      book: "A Journey to the Centre of the Earth",
      author: "Jules Verne",
      prefix:
        "Breakfast over, my uncle drew from his pocket a small notebook, intended for scientific observations. He consulted his instruments, and recorded:\n“Monday, July 1.\n“Chronometer, ",
      suffix:
        "; barometer, 297 in.; thermometer, 6° (43° F.). Direction, E.S.E.”\nThis last observation applied to the dark gallery, and was indicated by the compass.",
    },
    {
      time: "eight seventeen",
      book: "The Undomestic Goddess",
      author: "Sophie Kinsella",
      prefix:
        "Come on, I can't give up yet. I'll just sit here for a bit and enjoy the peace. I lean back and watch a little speckled bird pecking the ground nearby for a while. Then I look at my watch again: ",
      suffix: ". I can't do this.",
    },
  ],
  "08:19": [
    {
      time: "8.19",
      book: "The Terrible Privacy of Maxwell Sim",
      author: "Jonathan Coe",
      prefix:
        "I had arranged to meet the Occupational Health Officer at 10:30. I took the train from Watford Junction at ",
      suffix: " and arrived at London Euston seven minutes late, at 8.49.",
    },
  ],
  "08:20": [
    {
      time: "8:20",
      book: "Gravity's Rainbow",
      author: "Thomas Pynchon",
      prefix: "When the typewriters happen to pause (",
      suffix:
        " and other mythical hours), and there are no flights of American bombers in the sky, and the motor traffic's not too heavy in Oxford Street, you can hear winter birds cheeping outside, busy at the feeders the girls have put up.",
    },
  ],
  "08:23": [
    {
      time: "Twenty-three minutes past eight",
      book: "The Flowering of The Strange Orchid",
      author: "HG Wells",
      prefix: 'And then Wedderburn looked at his watch. "',
      suffix:
        ". I am going up by the quarter to twelve train, so that there is plenty of time. I think I shall wear my alpaca jacket - it is quite warm enough - and my grey felt hat and brown shoes. I suppose”",
    },
    {
      time: "8:23",
      book: "The Princess Bride",
      author: "William Goldman",
      prefix: "At ",
      suffix:
        " there seemed every chance of a lasting alliance starting between Florin and Guilder. At 8:24 the two nations were very close to war.",
    },
  ],
  "08:24": [
    {
      time: "8:24",
      book: "The Princess Bride",
      author: "William Goldman",
      prefix:
        "At 8:23 there seemed every chance of a lasting alliance starting between Florin and Guilder. At ",
      suffix: " the two nations were very close to war.",
    },
  ],
  "08:26": [
    {
      time: "twenty-six minutes past eight",
      book: "The Little Drummer Girl",
      author: "John Le Carre",
      prefix:
        "It exploded much later than intended, probably a good twelve hours later, at ",
      suffix:
        " on Monday morning. Several defunct wristwatches, the property of victims, confirmed the time. As with its predecessors over the last few months, there had been no warning.",
    },
  ],
  "08:27": [
    {
      time: "almost eight-thirty",
      book: "A Confederacy of Dunces",
      author: "John Kennedy Toole",
      prefix: "The lecture was to be given tomorrow, and it was now ",
      suffix: ".",
    },
  ],
  "08:28": [
    {
      time: "8.28",
      book: "The Riddle of the Sands",
      author: "Erskine Childers",
      prefix: "And at ",
      suffix:
        " on the following morning, with a novel chilliness about the upper lip, and a vast excess of strength and spirits, I was sitting in a third-class carriage, bound for Germany, and dressed as a young sea-man, in a pea-jacket, peaked cap, and comforter.",
    },
  ],
  "08:29": [
    {
      time: "8.29",
      book: "Engleby",
      author: "Sebastian Faulks",
      prefix: "At ",
      suffix:
        " I punched the front doorbell in Elgin Crescent. It was opened by a small oriental woman in a white apron. She showed me into a large, empty sitting room with an open fire and a couple of huge oil paintings.",
    },
  ],
  "08:30": [
    {
      time: "half past eight",
      book: "Harry Potter and the Philosopher's Stone",
      author: "JK Rowling",
      prefix: "At ",
      suffix:
        ", Mr. Dursley picked up his briefcase, pecked Mrs. Dursley on the cheek, and tried to kiss Dudley good-bye but missed, because Dudley was now having a tantrum and throwing his cereal at the walls.",
    },
    {
      time: "8:30",
      book: "Long Day's Journey Into Night",
      author: "Eugene O'Neill",
      prefix: "It is around ",
      suffix:
        ". Sunshine comes through the windows at right. As the curtain rises, the family has just finished breakfast.",
    },
    {
      time: "8:30 a.m.",
      book: "The Terrors of Ice and Darkness",
      author: "Christoph Ransmayr",
      prefix: "On July 25th, ",
      suffix:
        " the bitch Novaya dies whelping. At 10 o'clock she is lowered into her cool grave, at 7:30 that same evening we see our first floes and greet them wishing they were the last.",
    },
    {
      time: "eight-thirty",
      book: "A Confederacy of Dunces",
      author: "John Kennedy Toole",
      prefix: "The lecture was to be given tomorrow, and it was now almost ",
      suffix: ".",
    },
    {
      time: "eight-thirty",
      book: "Deaf Sentence",
      author: "David Lodge",
      prefix: "When he woke, at ",
      suffix:
        ", he was alone in the bedroom. He put on his dressing gown and put in his hearing aid and went into the living room.",
    },
  ],
  "08:32": [
    {
      time: "0832",
      book: "Infinite Jest",
      author: "David Foster Wallace",
      prefix:
        "'Does anybody know the time a little more exactly is what I'm wondering, Don, since Day doesn't.' Gately checks his cheap digital, head still hung over the sofa's arm. 'I got ",
      suffix: ":14, 15, 16, Randy.' ''ks a lot, D.G. man.'",
    },
    {
      time: "8.32 a.m.",
      book: "The Curious Incident Of The Dog In The Night-Time",
      author: "Mark Haddon",
      prefix: "",
      suffix: " Catch bus to schoo",
    },
  ],
  "08:35": [
    {
      time: "thirty-five minutes past eight",
      book: "Fruitfulness",
      author: "Emile Zola",
      prefix: "It was ",
      suffix:
        " by the big clock of the central building when Mathieu crossed the yard towards the office which he occupied as chief designer. For eight years he had been employed at the works where, after a brilliant and special course of study, he had made his beginning as assistant draughtsman when but nineteen years old, receiving at that time a salary of one hundred francs a month.",
    },
    {
      time: "8.35 a.m.",
      book: "Ulysses",
      author: "James Joyce",
      prefix:
        "Old gummy granny (thrusts a dagger towards Stephen's hand) Remove him, acushla. At ",
      suffix:
        " you will be in heaven and Ireland will be free (she prays) O good God take him!",
    },
  ],
  "08:37": [
    {
      time: "Eight thirty-seven",
      book: "Magic Bleeds",
      author: "Ilona Andrews",
      prefix: "",
      suffix:
        " am., Patrice Lane, Biohazard: The dog's clean. The Good Samaritan was a woman with an accent of some sort. Why haven't you called me",
    },
  ],
  "08:39": [
    {
      time: "8:39 A.M.",
      book: "Terminal Compromise",
      author: "Winn Schwartau",
      prefix: "Doug McGuire noticed the early hour, ",
      suffix:
        " on the one wall clock that gave Daylight Savings Time for the East Coast.",
    },
  ],
  "08:40": [
    {
      time: "8.40",
      book: "Around the world in eighty days",
      author: "Jules Verne",
      prefix: "At this moment the clock indicated ",
      suffix:
        ". 'Five minutes more,' said Andrew Stuart. The five friends looked at each other. One may surmise that their heart-beats were slightly accelereted, for, even for bold gamblers, the stake was a large one.'",
    },
    {
      time: "twenty minutes to nine",
      book: "Great Expectations",
      author: "Charles Dickens",
      prefix:
        "It was when I stood before her, avoiding her eyes, that I took note of the surrounding objects in detail, and saw that her watch had stopped at ",
      suffix:
        ", and that a clock in the room had stopped at twenty minutes to nine.",
    },
  ],
  "08:41": [
    {
      time: "forty-one minutes past eight",
      book: "Narrative of a Journey round the Dead Sea and in the Bible lands in 1850 and 1851",
      author: "Félicien de Saulcy",
      prefix: "By ",
      suffix:
        " we are five hundred yards from the water’s edge, and between our road and the foot of the mountain we descry the piled-up remains of a ruined tower.",
    },
  ],
  "08:43": [
    {
      time: "eight forty-three",
      book: "A Time to Kill",
      author: "John Grisham",
      prefix:
        '"You understand this tape recorder is on?" "Uh huh" "And it\'s Wednesday, May 15, at ',
      suffix: ' in the mornin\'." "If you say so"',
    },
    {
      time: "8.43 a.m.",
      book: "The Curious Incident Of The Dog In The Night-Time",
      author: "Mark Haddon",
      prefix: "",
      suffix: " Go past tropical fish sho",
    },
  ],
  "08:44": [
    {
      time: "eight forty-four",
      book: "The Secret Miracle",
      author: "Jorge Luis Borges",
      prefix:
        "Several soldiers - some with their uniforms unbuttoned - were looking over a motorcycle, arguing about it. The sergeant looked at his watch; it was ",
      suffix:
        ". They had to wait until nine. Hladik, feeling more insignificant than ill-fortuned, sat down on a pile of firewood.",
    },
  ],
  "08:45": [
    {
      time: "8:45",
      book: "Dreams of leaving",
      author: "Rupert Thomson",
      prefix: "He paid the waitress and left the café. It was ",
      suffix:
        ". The sun pressed against the inside of a thin layer of cloud. He unbuttoned his jacket as he hurried down Queensway. His mind, unleashed, sprang forwards.",
    },
  ],
  "08:47": [
    {
      time: "8.47",
      book: "Dirk Gently's Holistic Detective Agency",
      author: "Douglas Adams",
      prefix: "\"Just on my way to the cottage. It's, er, ..",
      suffix: '. Bit misty on the roads....."',
    },
    {
      time: "8.47",
      book: "Dirk Gently's Holistic Detective Agency",
      author: "Douglas Adams",
      prefix: "",
      suffix: ". Bit misty on the road",
    },
  ],
  "08:49": [
    {
      time: "8.49",
      book: "The Terrible Privacy of Maxwell Sim",
      author: "Jonathan Coe",
      prefix:
        "I had arranged to meet the Occupational Health Officer at 10:30. I took the train from Watford Junction at 8.19 and arrived at London Euston seven minutes late, at ",
      suffix: ".",
    },
  ],
  "08:50": [
    {
      time: "ten to nine",
      book: "The Chestnut Tree",
      author: "V.S. Pritchett",
      prefix: "At ",
      suffix:
        " the clerks began to arrive.When they had hung up their coats and hates they came to the fireplace and stood warming themselves. If there was no fire, they stood there all the same",
    },
    {
      time: "8:50",
      book: "The Ask",
      author: "Sam Lipsyte",
      prefix: "It was ",
      suffix:
        ' in the morning and Bernie and I were alone on an Astoria side street, not far from a sandwich shop that sold a sopressatta sub called "The Bypass". I used to eat that sandwich weekly, wash it down with espresso soda, smoke a cigarette, go for a jog. Now I was too near the joke to order the sandwich, and my son\'s preschool in the throes of doctrinal schism.',
    },
    {
      time: "ten minutes to nine",
      book: "The Radetzky March",
      author: "Joseph Roth",
      prefix: "Punctually at ",
      suffix:
        ", a quarter hour after early mass, the boy stood in his Sunday uniform outside his father's door.",
    },
  ],
  "08:51": [
    {
      time: "8.51 a.m.",
      book: "The Curious Incident of the Dog in the Night-time",
      author: "Mark Haddon",
      prefix: "",
      suffix: " Arrive at schoo",
    },
  ],
  "08:52": [
    {
      time: "8.52am.",
      book: "Extremely Loud and Incredibly Close",
      author: "Jonathan Safran Foer",
      prefix: "Message one. Tuesday, ",
      suffix: " Is anybody there? Hello?",
    },
  ],
  "08:54": [
    {
      time: "nearly nine o’clock",
      book: "Three Men in a Boat",
      author: "Jerome K Jerome",
      prefix:
        "It was Mrs. Poppets that woke me up next morning. She said: “Do you know that it’s ",
      suffix:
        ", sir?” “Nine o’ what?” I cried, starting up. “Nine o’clock,” she replied, through the keyhole. “I thought you was a- oversleeping yourselves.”",
    },
  ],
  "08:55": [
    {
      time: "five minutes to nine",
      book: "The Radetzky March",
      author: "Joseph Roth",
      prefix: "At ",
      suffix:
        ', Jacques, in his gray butler\'s livery, came down the stairs and said, "Young master, your Herr Papá is coming."',
    },
    {
      time: "five minutes to nine",
      book: "Three Men in a Boat",
      author: "Jerome K Jerome",
      prefix: "George pulled out his watch and looked at it: it was ",
      suffix: "!",
    },
  ],
  "08:56": [
    {
      time: "nearly nine o'clock",
      book: "Burmese Days",
      author: "George Orwell",
      prefix: "It was ",
      suffix: " and the sun was fiercer every minute.'",
    },
  ],
  "08:57": [
    {
      time: "three minutes before nine",
      book: "The Paradise Mystery",
      author: "JS Fletcher",
      prefix:
        "You'll have to hurry. Many a long year before that, in one of the bygone centuries, a worthy citizen of Wrychester, Martin by name, had left a sum of money to the Dean and Chapter of the Cathedral on condition that as long as ever the Cathedral stood, they should cause to be rung a bell from its smaller bell-tower for ",
      suffix: " o'clock every morning, all the year round.",
    },
  ],
  "08:58": [
    {
      time: "two minutes of nine",
      book: "The Getaway",
      author: "Jim Thompson",
      prefix: "It was ",
      suffix:
        " now - two minutes before the bombs were set to explode - and three or four people were gathered in front of the bank waiting for it to open.",
    },
  ],
  "08:59": [
    {
      time: "8:59",
      book: "Sophie's World",
      author: "Jostein Gaarder",
      prefix:
        "She had been lying in bed reading about Sophie and Alberto's conversation on Marx and had fallen asleep. The reading lamp by the bed had been on all night. The green glowing digits on her desk alarm clock showed ",
      suffix: ".",
    },
  ],
  "09:00": [
    {
      time: "nine o'clock",
      book: "A Confederacy of Dunces",
      author: "John Kennedy Toole",
      prefix: "'I could never get all the way down there before ",
      suffix: ".'",
    },
    {
      time: "nine o'clock",
      book: "A Confederacy of Dunces",
      author: "John Kennedy Toole",
      prefix: "'Look. Ignatius. I'm beat. I've been on the road since ",
      suffix: " yesterday morning.'",
    },
    {
      time: "nine",
      book: "The Pickwick Papers",
      author: "Charles Dickens",
      prefix:
        "On the third morning after their arrival, just as all the clocks in the city were striking ",
      suffix:
        " individually, and somewhere about nine hundred and ninety-nine collectively, Sam was taking the air in George Yard, when a queer sort of fresh painted vehicle drove up, out of which there jumped with great agility, throwing the reins to a stout man who sat beside him, a queer sort of gentleman, who seemed made for the vehicle, and the vehicle for him.",
    },
    {
      time: "9:00 am",
      book: "A Fraction of the Whole",
      author: "Steve Toltz",
      prefix: "14 June ",
      suffix: " woke up",
    },
    {
      time: "9.00 a.m.",
      book: "The Curious Incident of the Dog in the Night-time",
      author: "Mark Haddon",
      prefix: "",
      suffix: " School assembl",
    },
    {
      time: "nine",
      book: "The Radetzky March",
      author: "Joseph Roth",
      prefix: "A fly buzzed, the wall clock began to strike. After the ",
      suffix:
        ' golden strokes faded, the district captain began. "How is Herr Colonel Marek?" "Thank you, Papá, he\'s fine." "Still weak in geometry?" "Thank you, Papá, a little better." "Read any books?" "Yessir, Papá."',
    },
    {
      time: "nine o' clock",
      book: "A Single Pebble",
      author: "John Hershey",
      prefix: "As ",
      suffix:
        " was left behind, the preposterousness of the delay overwhelmed me, and I went in a kind of temper to the owner and said that I thought he should sign on another cook and weigh spars and be off.",
    },
    {
      time: "nine o'clock",
      book: "The Great Gatsby",
      author: "F. Scott Fitzgerald",
      prefix: "At ",
      suffix:
        ", one morning late in July, Gatsby's gorgeous car lurched up the rocky drive to my door and gave out a burst of melody from its three-noted horn",
    },
    {
      time: "at nine",
      book: "The Warden",
      author: "Anthony Trollope",
      prefix: "He was at breakfast ",
      suffix:
        ', and for the twentieth time consulted his "Bradshaw," to see at what earliest hour Dr. Grantly could arrive from Barchester.',
    },
    {
      time: "nine o'clock",
      book: "Alice's Adventures in Wonderland",
      author: "Lewis Carroll",
      prefix:
        "He won't stand beating. Now, if you only kept on good terms with him, he'd do almost anything you liked with the clock. For instance, suppose it were ",
      suffix:
        " in the morning, just time to begin lessons: you'd only have to whisper a hint to Time, and round goes the clock in a twinkling! Half-past one, time for dinner!",
    },
    {
      time: "nine o'clock",
      book: "The Remains of the Day",
      author: "Kazuo Ishiguro",
      prefix: "It was around ",
      suffix:
        " that I crossed the border into Cornwall. This was at least three hours before the rain began and the clouds were still all of a brilliant white. In fact, many of the sights that greeted me this morning were among the most charming I have so far encountered. It was unfortunate, then, that I could not for much of the time give to them the attention they warranted; for one may as well declare it, one was in a condition of some preoccupation with the thought that - barring some unseen complication - one would be meeting Miss Kenton again before the day's end.",
    },
    {
      time: "At nine",
      book: "Death in Venice",
      author: "Thomas Mann",
      prefix:
        "Opening his window, Aschenbach thought he could smell the foul stench of the lagoon. A sudden despondency came over him. He considered leaving then and there. Once, years before, after weeks of a beautiful spring, he had been visited by this sort of weather and it so affected his health he had been obliged to flee. Was not the same listless fever setting in? The pressure in the temples, the heavy eyelids? Changing hotels again would be a nuisance, but if the wind failed to shift he could not possibly remain here. To be on the safe side, he did not unpack everything. ",
      suffix:
        " he went to breakfast in the specially designated buffet between the lobby and the dining room.",
    },
    {
      time: "9.00am",
      book: "Jesus' Son",
      author: "Denis Johnson",
      prefix:
        "Sometimes what I wouldn't give to have us sitting in a bar again at ",
      suffix: " telling lies to one another, far from God.",
    },
    {
      time: "nine",
      book: "Romeo and Juliet",
      author: "Shakespeare",
      prefix: "The clock struck ",
      suffix:
        " when I did send the nurse; In half an hour she promised to return. Perchance she cannot meet him: that's not so.",
    },
    {
      time: "nine",
      book: "The Waste Land",
      author: "T S Eliot",
      prefix:
        "To where Saint Mary Woolnoth kept the hours With a dead sound on the final stroke of ",
      suffix: ".",
    },
    {
      time: "nine",
      book: "The Waste Land",
      author: "T S Eliot",
      prefix:
        "Unreal City, Under the brown fog of a winter dawn, A crowd flowed over London Bridge, so many, I had not thought death had undone so many. Sighs, short and infrequent, were exhaled, And each man fixed his eyes before his feet. Flowed up the hill and down King William Street, To where Saint Mary Woolnoth kept the hours With a dead sound on the final stroke of ",
      suffix: ".",
    },
  ],
  "09:01": [
    {
      time: "9:01am",
      book: "A Fraction of the Whole",
      author: "Steve Toltz",
      prefix: "",
      suffix: " lay in bed, staring at ceiling",
    },
  ],
  "09:02": [
    {
      time: "9:02am",
      book: "A Fraction of the Whole",
      author: "Steve Toltz",
      prefix: "",
      suffix: " lay in bed, staring at ceiling",
    },
  ],
  "09:03": [
    {
      time: "9:03am",
      book: "A Fraction of the Whole",
      author: "Steve Toltz",
      prefix: "",
      suffix: " lay in bed, staring at ceiling",
    },
    {
      time: "three minutes past nine",
      book: "The Lottie Project",
      author: "Jacqueline Wilson",
      prefix:
        "This isn't a very good start to the new school year.\" I stared at her. What was she talking about? Why was she looking at her watch? I wasn't late. Okay, the school bell had rung as I was crossing the playground, but you always get five minutes to get to your classroom. \"It's ",
      suffix: '," Miss Beckworth announced. "You\'re late."',
    },
  ],
  "09:04": [
    {
      time: "9:04am",
      book: "A Fraction of the Whole",
      author: "Steve Toltz",
      prefix: "",
      suffix: " lay in bed, staring at ceilin",
    },
    {
      time: "9.04",
      book: "The Great Train Robbery",
      author: "Michael Crichton",
      prefix:
        "In the light of a narrow-beam lantern, Pierce checked his watch. It was ",
      suffix: ".",
    },
  ],
  "09:05": [
    {
      time: "9:05am",
      book: "A Fraction of the Whole",
      author: "Steve Toltz",
      prefix: "",
      suffix: " lay in bed, staring at ceilin",
    },
    {
      time: "9:05",
      book: "The Voices of Time",
      author: "JG Ballard",
      prefix:
        "Kaldren pursues me like luminescent shadow. He has chalked up on the gateway '96,688,365,498,702'. Should confuse the mail man. Woke ",
      suffix: ". To sleep 6:36.",
    },
    {
      time: "9:05 a.m.",
      book: "Twenties Girl",
      author: "Sophie Kinsella",
      prefix:
        "The tour of the office doesn't take that long. In fact, we're pretty much done by ",
      suffix:
        " … and even though it's just a room with a window and a pin board and two doors and two desks... I can't help feeling a buzz as I lead them around. It's mine. My space. My company.",
    },
    {
      time: "9:05 a.m.",
      book: "Twenties Girl",
      author: "Sophie Kinsella",
      prefix:
        "The tour of the office doesn't take that long. In fact, we're pretty much done by ",
      suffix:
        " Ed looks at everything twice and says it's all great, and gives me a list of contacts who might be helpful, then has to leave for his own office.",
    },
  ],
  "09:06": [
    {
      time: "9:06am",
      book: "A Fraction of the Whole",
      author: "Steve Toltz",
      prefix: "",
      suffix: " lay in bed, staring at ceilin",
    },
  ],
  "09:07": [
    {
      time: "9:07am",
      book: "A Fraction of the Whole",
      author: "Steve Toltz",
      prefix: "",
      suffix: " lay in bed, staring at ceilin",
    },
    {
      time: "9:07",
      book: "The Stone Diaries",
      author: "Carol Shields",
      prefix: "It was a sparkling morning, ",
      suffix:
        " by the clock when Mrs. Flett stepped aboard the Imperial Limited at the Tyndall station, certain that her life was ruined, but managing, through an effort of will, to hold herself erect and to affect an air of preoccupation and liveliness.",
    },
  ],
  "09:08": [
    {
      time: "9.08am",
      book: "A Fraction of the Whole",
      author: "Steve Toltz",
      prefix: "",
      suffix: " rolled over onto left side",
    },
  ],
  "09:09": [
    {
      time: "9.09am",
      book: "A Fraction of the Whole",
      author: "Steve Toltz",
      prefix: "",
      suffix: " lay in bed, staring at wall",
    },
  ],
  "09:10": [
    {
      time: "9.10am",
      book: "A Fraction of the Whole",
      author: "Steve Toltz",
      prefix: "",
      suffix: " lay in bed, staring at wall",
    },
  ],
  "09:11": [
    {
      time: "9:11am",
      book: "A Fraction of the Whole",
      author: "Steve Toltz",
      prefix: "",
      suffix: " lay in bed, staring at wal",
    },
  ],
  "09:12": [
    {
      time: "9.12am",
      book: "A Fraction of the Whole",
      author: "Steve Toltz",
      prefix: "",
      suffix: " lay in bed, staring at wall",
    },
  ],
  "09:13": [
    {
      time: "9:13am",
      book: "A Fraction of the Whole",
      author: "Steve Toltz",
      prefix: "",
      suffix: " lay in bed, staring at wal",
    },
    {
      time: "9:13 A.M.",
      book: "Mistaken Identity",
      author: "Lisas Scottoline",
      prefix:
        "She tucked the phone in the crook of her neck and thumbed hurriedly through her pink messages. Dr. Provetto, at ",
      suffix: "",
    },
  ],
  "09:14": [
    {
      time: "9.14am",
      book: "A Fraction of the Whole",
      author: "Steve Toltz",
      prefix: "",
      suffix: " lay in bed, staring at wall",
    },
  ],
  "09:15": [
    {
      time: "0915",
      book: "The Hunt for Red October",
      author: "Tom Clancy",
      prefix:
        '"Great!" Jones commented. "I\'ve never seen it do that before. That\'s all right. Okay." Jones pulled a handful of pencils from his back pocket. "Now, I got the contact first at ',
      suffix: ' or so, and the bearing was about two-six-nine."',
    },
    {
      time: "9:15am",
      book: "A Fraction of the Whole",
      author: "Steve Toltz",
      prefix: "",
      suffix: " doubled over pillow, sat up to see out windo",
    },
    {
      time: "9.15 a.m.",
      book: "The Curious Incident of the Dog in the Night-time",
      author: "Mark Haddon",
      prefix: "",
      suffix: " First morning clas",
    },
    {
      time: "quarter past nine",
      book: "Miss Pettigrew Lives For a Day",
      author: "Winifred Watson",
      prefix:
        "Miss Pettigrew pushed open the door of the employment agency and went in as the clock struck a ",
      suffix: ".",
    },
  ],
  "09:16": [
    {
      time: "9.16am",
      book: "A Fraction of the Whole",
      author: "Steve Toltz",
      prefix: "",
      suffix: " sat in bed, staring out window",
    },
  ],
  "09:17": [
    {
      time: "9.17am",
      book: "A Fraction of the Whole",
      author: "Steve Toltz",
      prefix: "",
      suffix: " sat in bed, staring out window",
    },
  ],
  "09:18": [
    {
      time: "9.18am",
      book: "A Fraction of the Whole",
      author: "Steve Toltz",
      prefix: "",
      suffix: " sat in bed, staring out window",
    },
  ],
  "09:19": [
    {
      time: "9.19am",
      book: "A Fraction of the Whole",
      author: "Steve Toltz",
      prefix: "",
      suffix: " sat in bed, staring out window",
    },
  ],
  "09:20": [
    {
      time: "nine-twenty",
      book: "Girl, Interrupted",
      author: "Susanna Kaysen",
      prefix:
        "I'll compromise by saying that I left home at eight and spent an hour travelling to a nine o'clock appointment. Twenty minutes later is ",
      suffix: ".",
    },
    {
      time: "twenty minutes past nine",
      book: "Uncle Fred in the Springtime",
      author: "P.G. Wodehouse",
      prefix: "At ",
      suffix:
        ", the Duke of Dunstable, who had dined off a tray in his room, was still there, waiting for his coffee and liqueur.",
    },
    {
      time: "9.20",
      book: "Red Dog",
      author: "Louis de Bernieres",
      prefix: "The following morning at ",
      suffix:
        " Mr Cribbage straightened his greasy old tie, combed his Hitler moustache and arranged the few strands of his hair across his bald patch.",
    },
  ],
  "09:21": [
    {
      time: "nine twenty-one",
      book: "This is Life",
      author: "Dan Rhodes",
      prefix: "It was ",
      suffix: ". With one minute to go, there was no sign of Herbert's mother.",
    },
  ],
  "09:22": [
    {
      time: "nine twenty-two",
      book: "This is Life",
      author: "Dan Rhodes",
      prefix:
        "No more throwing stones at him, and I'll see you back here exactly one week from now. She looked at her watch. 'At ",
      suffix: " next Wednesday.'",
    },
  ],
  "09:23": [
    {
      time: "9.23",
      book: "Ulysses",
      author: "James Joyce",
      prefix: "",
      suffix: ". What possessed me to buy this comb",
    },
  ],
  "09:24": [
    {
      time: "9.24",
      book: "Ulysses",
      author: "James Joyce",
      prefix: "",
      suffix:
        " I'm swelled after that cabbage. A speck of dust on the patent leather of her boot",
    },
  ],
  "09:25": [
    {
      time: "nine twenty-five",
      book: "The Forgotten Waltz",
      author: "Anne Enright",
      prefix: "A man I would cross the street to avoid at nine o'clock - by ",
      suffix:
        " I wanted to fuck him until he wept. My legs trembled with it. My voice floated out of my mouth when I opened it to speak. The glass wall of the meeting room was huge and suddenly too transparent.",
    },
  ],
  "09:27": [
    {
      time: "twenty-seven minutes past nine",
      book: "Sinister Street",
      author: "Compton Mackenzie",
      prefix: "From twenty minutes past nine until ",
      suffix:
        ", from twenty-five minutes past eleven until twenty-eight minutes past eleven, from ten minutes to three until two minutes to three the heroes of the school met in a large familiarity whose Olympian laughter awed the fearful small boy that flitted uneasily past and chilled the slouching senior that rashly paused to examine the notices in assertion of an unearned right.",
    },
  ],
  "09:28": [
    {
      time: "twenty-eight minutes past nine",
      book: "Lord Raingo",
      author: "Arnold Bennett",
      prefix:
        '"This clock right?" he asked the butler in the hall. "Yes, sir." The clock showed ',
      suffix:
        '. "The clocks here have to be right, sir," the butler added with pride and a respectful humour, on the stairs.',
    },
    {
      time: "twenty-eight minutes past nine",
      book: "Lord Raingo",
      author: "Arnold Bennett",
      prefix:
        'He entered No. 10 for the first time, he who had sat on the Government benches for eight years and who had known the Prime Minister from youth up. "This clock right?" he asked the butler in the hall. "Yes, sir." The clock showed ',
      suffix:
        '. "The clocks here have to be right, sir," the butler added with pride and a respectful humour, on the stairs.',
    },
  ],
  "09:30": [
    {
      time: "half-past nine",
      book: "A watcher by the dead",
      author: "Ambrose Bierce",
      prefix: "he looked at his watch; it was ",
      suffix: "",
    },
    {
      time: "nine-thirty",
      book: "Revolutionary Road",
      author: "Richard Yates",
      prefix: "It was ",
      suffix:
        ". In another ten minutes she would turn off the heat; then it would take a while for the water to cool. In the meantime there was nothing to do but wait. “Have you thought it through April?” Never undertake to do a thing until you’ve –“ But she needed no more advice and no more instruction. She was calm and quiet now with knowing what she had always known, what neither her parents not Aunt Claire not Frank nor anyone else had ever had to teach her: that if you wanted to do something absolutely honest, something true, it always turned out to be a thing that had to be done alone.",
    },
    {
      time: "nine-thirty",
      book: "Trumpet",
      author: "Jackie Kay",
      prefix: "The body came in at ",
      suffix:
        " this morning. One of Holding's men went to the house and collected it. There was nothing particularly unusual about the death. The man had had a fear of hospitals and had died at home, being cared for more than adequately by his devoted wife.",
    },
    {
      time: "9.30",
      book: "Independence Day",
      author: "Richard Ford",
      prefix:
        "Up the welcomingly warm morning hill we trudge, side by each, bound finally for the Hall of Fame. It's ",
      suffix: ", and time is in fact a-wastin'.",
    },
  ],
  "09:32": [
    {
      time: "9.32",
      book: "Three Men in a Boat",
      author: "Jerome K Jerome",
      prefix:
        "He said he couldn't say for certain of course, but that he rather thought he was. Anyhow, if he wasn't the 11.5 for Kingston, he said he was pretty confident he was the ",
      suffix:
        " for Virginia Water, or the 10 a.m. express for the Isle of Wight, or somewhere in that direction, and we should all know when we got there.",
    },
    {
      time: "nine-thirty-two",
      book: "Wifey",
      author: "Judy Blume",
      prefix: "Sandy barely made the ",
      suffix:
        " and found a seat in no-smoking. She'd been looking forward to this visit with Lisbeth. They hadn't seen each other in months, not since January, when Sandy had returned from Jamaica. And on that day Sandy was sporting a full-blown herpes virus on her lower lip.",
    },
  ],
  "09:33": [
    {
      time: "thirty-three minutes past nine",
      book: "The Toilers of the Sea",
      author: "Victor Hugo",
      prefix:
        "Next, he remembered that the morrow of Christmas would be the twenty-seventh day of the moon, and that consequently high water would be at twenty-one minutes past three, the half-ebb at a quarter past seven, low water at ",
      suffix: ", and half flood at thirty-nine minutes past twelve.",
    },
  ],
  "09:35": [
    {
      time: "Nine-thirty-five",
      book: "The Memory of Love",
      author: "Aminatta Forna",
      prefix: "",
      suffix:
        ". He really must be gone. The bird is no longer feeding but sitting at the apex of a curl of razor wire",
    },
    {
      time: "Nine-thirty-five",
      book: "The Memory of Love",
      author: "Aminatta Forna",
      prefix: "",
      suffix:
        ". He really must be gone. The bird is no longer feeding but sitting at the apex of a curl of razor wire",
    },
  ],
  "09:36": [
    {
      time: "9:36",
      book: "The Undomestic Goddess",
      author: "Sophie Kinsella",
      prefix:
        "I grab a pen and the pad of paper by the phone and start scribbling a list for the day. I have an image of myself moving smoothly from task to task, brush in one hand, duster in the other, bringing order to everything. Like Mary Poppins. 9:30-",
      suffix:
        " Make Geigers' bed 9:36-9:42 Take laundry out of machine and put in dryer 9:42-10:00 Clean bathrooms I get to the end and read it over with a fresh surge of optimism. At this rate I should be done easily by lunchtime. 9:36 Fuck. I cannot make this bed. Why won't this sheet lie flat? 9:42 And why do they make mattresses so heavy?",
    },
    {
      time: "9.36am.",
      book: "Bridget Jones Diary",
      author: "Helen Fielding",
      prefix: "Monday February 6th. '",
      suffix:
        " Oh god, Oh god. Maybe he's fallen in love in New York and stayed there'.",
    },
  ],
  "09:37": [
    {
      time: "thirty-seven minutes past nine",
      book: "Around the World in 80 days",
      author: "Jules Verne",
      prefix:
        "It comprised all that was required of the servant, from eight in the morning, exactly at which hour Phileas Fogg rose, till half-past eleven, when he left the house for the Reform Club - all the details of service, the tea and toast at twenty-three minutes past eight, the shaving-water at ",
      suffix: ", and the toilet at twenty minutes before ten.",
    },
  ],
  "09:40": [
    {
      time: "twenty minutes before ten",
      book: "Around the World in 80 days",
      author: "Jules Verne",
      prefix:
        "It comprised all that was required of the servant, from eight in the morning, exactly at which hour Phileas Fogg rose, till half-past eleven, when he left the house for the Reform Club—all the details of service, the tea and toast at twenty-three minutes past eight, the shaving-water at thirty-seven minutes past nine, and the toilet at ",
      suffix: ".",
    },
    {
      time: "9:40",
      book: "The Voices of Time",
      author: "JG Ballard",
      prefix:
        "Must have the phone disconnected. Some contractor keeps calling me up about payment for 50 bags of cement he claims I collected ten days ago. Says he helped me load them onto a truck himself. I did drive Whitby's pick-up into town but only to get some lead screening. What does he think I'd do with all that cement? Just the sort of irritating thing you don't expect to hang over your final exit. (Moral: don't try too hard to forget Eniwetok.) Woke ",
      suffix: ". To sleep 4:15.",
    },
  ],
  "09:42": [
    {
      time: "9:42",
      book: "The Undomestic Goddess",
      author: "Sophie Kinsella",
      prefix:
        "I grab a pen and the pad of paper by the phone and start scribbling a list for the day. I have an image of myself moving smoothly from task to task, brush in one hand, duster in the other, bringing order to everything. Like Mary Poppins. 9:30-9:36 Make Geigers' bed 9:36-",
      suffix:
        " Take laundry out of machine and put in dryer 9:42-10:00 Clean bathrooms I get to the end and read it over with a fresh surge of optimism. At this rate I should be done easily by lunchtime. 9:36 Fuck. I cannot make this bed. Why won't this sheet lie flat? 9:42 And why do they make mattresses so heavy?",
    },
  ],
  "09:45": [
    {
      time: "9.45",
      book: "On Her Majesty's Secret Service",
      author: "Ian Fleming",
      prefix: "9.15, 9.30, ",
      suffix:
        ", 10! Bond felt the excitement ball up inside him like cat's fur.",
    },
  ],
  "09:47": [
    {
      time: "9.47am.",
      book: "Bridget Jones Diary",
      author: "Helen Fielding",
      prefix: "Monday February 6th. '",
      suffix: " Or gone to Las Vegas and got married'.",
    },
  ],
  "09:50": [
    {
      time: "9.50am.",
      book: "Bridget Jones Diary",
      author: "Helen Fielding",
      prefix: "",
      suffix: " Hmmm. Think will go inspect make-up in case he does come i",
    },
    {
      time: "Ten minutes to ten.",
      book: "The Law and the Lady",
      author: "Wilkie Collins",
      prefix: "",
      suffix:
        ' "I had just time to hide the bottle (after the nurse had left me) when you came into my room.',
    },
  ],
  "09:52": [
    {
      time: "9:52",
      book: "Call for the Dead",
      author: "John le Carre",
      prefix: '"She caught the ',
      suffix:
        ' to Victoria. I kept well clear of her on the train and picked her up as she went through the barrier. Then she took a taxi to Hammersmith." "A taxi?" Smiley interjected. "She must be out of her mind."',
    },
  ],
  "09:53": [
    {
      time: "seven minutes to ten",
      book: "Miss Pettigrew Lives for a Day",
      author: "Winifred Watson",
      prefix:
        "Miss Pettigrew went to the bus-stop to await a bus. She could not afford the fare, but she could still less afford to lose a possible situation by being late. The bus deposited her about five minutes' walk from Onslow Mansions, an at ",
      suffix: " precisely she was outside her destination.",
    },
  ],
  "09:54": [
    {
      time: "9:54",
      book: "The Undomestic Goddess",
      author: "Sophie Kinsella",
      prefix: "",
      suffix:
        " This is sheer torture. My arms have never ached so much in my entire life. The blankets weigh a ton, and the sheets won't go straight and I have no idea how to do the wretched corners. How do chambermaids do it",
    },
  ],
  "09:55": [
    {
      time: "five to ten",
      book: "The Undomestic Goddess",
      author: "Sophie Kinsella",
      prefix: "At ",
      suffix:
        " I'm ready in the hall. Nathaniel's mother's house is nearby but apparently tricky to find, so the plan is to meet here and he'll walk me over. I check my reflection in the hall mirror and wince. The streak of bleach in my hair is as obvious as ever. Am I really going out in public like this?",
    },
    {
      time: "five minutes to ten",
      book: "The Count of Monte Cristo",
      author: "Alexandre Dumas",
      prefix:
        'Good-morning, Lucien, good-morning, said Albert; "your punctuality really alarms me. What do I say? punctuality! You, whom I expected last, you arrive at ',
      suffix:
        ', when the time fixed was half-past! Has the ministry resigned?"',
    },
  ],
  "09:58": [
    {
      time: "around ten o'clock",
      book: "Catcher in the Rye",
      author: "J.D. Salinger",
      prefix: "I didn't sleep too long, because I think it was only ",
      suffix:
        " when I woke up. I felt pretty hungry as soon as I had a cigarette. The last time I'd eaten was those two hamburgers I had with Brossard and Ackley when we went in to Agerstown to the movies. That was a long time ago. It seemed like fifty years ago.",
    },
  ],
  "09:59": [
    {
      time: "One minute to ten.",
      book: "The Ragged Trouserred Philanthropists",
      author: "Robert Tressell",
      prefix: "",
      suffix:
        " With a heavy heart Bert watched the clock. His legs were still aching very badly. He could not see the hands of the clock moving, but they were creeping on all the same",
    },
  ],
  "10:00": [
    {
      time: "ten o'clock",
      book: "Tristram Shandy",
      author: "Laurence Sterne",
      prefix:
        "––In assaying to put on his regimental coat and waistcoat, my uncle Toby found the same objection in his wig, ––so that went off too: ––So that with one thing and what with another, as always falls out when a man is in the most haste, ––'twas ",
      suffix:
        ", which was half an hour later than his usual time before my uncle Toby sallied out.",
    },
    {
      time: "an hour ago since it was nine",
      book: "As You Like It",
      author: "William Shakespeare",
      prefix: "’Tis but ",
      suffix: ", And after one hour more ‘twill be eleven.",
    },
    {
      time: "Ten",
      book: "Orlando",
      author: "Virginia Woolf",
      prefix:
        "For some seconds the light went on becoming brighter and brighter, and she saw everything more and more clearly and the clock ticked louder and louder until there was a terrific explosion right in her ear. Orlando leapt as if she had been violently struck on the head. ",
      suffix:
        " times she was struck. In fact it was ten o'clock in the morning. It was the eleventh of October. It was 1928. It was the present moment.",
    },
    {
      time: "10:00",
      book: "The Girl with the Dragon Tattoo",
      author: "Stieg Larsson",
      prefix:
        "The trial was irretrievably over; everything that could be said had been said, but he had never doubted that he would lose. The written verdict was handed down at ",
      suffix:
        " on Friday morning, and all that remained was a summing up from the reporters waiting in the corridor outside the district court.",
    },
    {
      time: "10 am",
      book: "Kafka on the shore",
      author: "Haruki Murakami",
      prefix:
        "According to military records no US bombers or any other kind of aircraft were flying over that region at the time, that is around ",
      suffix: " on November 7,1944.",
    },
    {
      time: "ten o'clock",
      book: "Of Mice And Men",
      author: "John Steinbeck",
      prefix: "At about ",
      suffix:
        " in the morning the sun threw a bright dust-laden bar through one of the side windows, and in and out of the beam flies shot like rushing stars.",
    },
    {
      time: "ten o' clock",
      book: "The Medusa Frequency",
      author: "Russell Hoban",
      prefix:
        "I went to bed and the next thing I knew I was awake again and it was getting on for ",
      suffix: " in the morning. Ring, ring, said the telephone, ring, ring.",
    },
    {
      time: "ten o'clock",
      book: "Northanger Abbey",
      author: "Jane Austen",
      prefix:
        "If Wednesday should ever come! It did come, and exactly when it might be reasonably looked for. It came - it was fine - and Catherine trod on air. By ",
      suffix:
        ", the chaise and four conveyed the two from the abbey; and, after an agreeable drive of almost twenty miles, they entered Woodston, a large and populous village, in a situation not unpleasant.",
    },
    {
      time: "ten",
      book: "Richard III",
      author: "William Shakespeare",
      prefix:
        "King Richard: Well, but what's o'clock? Buckingham: Upon the stroke of ",
      suffix: ".",
    },
    {
      time: "10 o’clock",
      book: "The Diary of Samuel Pepys",
      author: "Samuel Pepys",
      prefix:
        "Monday 30 March 1668 Up betimes, and so to the office, there to do business till about ",
      suffix: "",
    },
    {
      time: "10 o'clock",
      book: "The Terrors of Ice and Darkness",
      author: "Christoph Ransmayr",
      prefix: "On July 25th, 8:30 a.m. the bitch Novaya dies whelping. At ",
      suffix:
        " she is lowered into her cool grave, at 7:30 that same evening we see our first floes and greet them wishing they were the last.",
    },
    {
      time: "Ten-thirty",
      book: "An Obedient Father",
      author: "Akhil Sharma",
      prefix:
        "The pundit sighed. 'Only a fool like me would leave his door open when a riot can occur at any moment, and only a fool like me would say yes to you,' he said. 'What time?' Just his head was sticking out of the partially opened door. The money from blessing the ice-cream factory must have dulled his desire for work, I thought. 'Ten.' '",
      suffix: ".' Without another word, he closed the door.",
    },
    {
      time: "ten o' clock",
      book: "The Greeks have a word for it",
      author: "Barry Unsworth",
      prefix:
        "The Saturday immediately preceding the examinations was a very busy day for Kennedy. At ",
      suffix:
        " he was entering Willey's room; the latter had given him a key and left the room vacant by previous arrangement - in fact he had taken Olivia on another house hunting trip.",
    },
    {
      time: "at ten",
      book: "Dubliners",
      author: "James Joyce",
      prefix:
        "The summer holidays were near at hand when I made up my mind to break out of the weariness of school-life for one day at least. With Leo Dillon and a boy named Mahoney I planned a day's mitching. Each of us saved up sixpence. We were to meet ",
      suffix: " in the morning on the Canal Bridge.",
    },
    {
      time: "10:00",
      book: "The girl with the dragon tattoo",
      author: "Stieg Larsson",
      prefix: "The written verdict was handed down at ",
      suffix:
        " on Friday morning, and allthat remained was a summing up from the reporters waiting in the corridor outside the district court.",
    },
  ],
  "10:01": [
    {
      time: "about ten o'clock",
      book: "Of Mice And Men",
      author: "John Steinbeck",
      prefix: "At ",
      suffix:
        " in the morning the sun threw a bright dust-laden bar through one of the side windows, and in and out of the beam flies shot like rushing stars.'",
    },
  ],
  "10:02": [
    {
      time: "two minutes after ten",
      book: "The Daemon Lover",
      author: "Shirley Jackson",
      prefix: "It was ",
      suffix:
        "; she was not satisfied with her clothes, her face, her apartment. She heated the coffee again and sat down in the chair by the window. Can't do anything more now, she thought, no sense trying to improve anything the last minute.",
    },
  ],
  "10:03": [
    {
      time: "10.03",
      book: "“Vanilla-Bright like Eminem” from The Farenheit Twins",
      author: "Michel Faber",
      prefix: "It's ",
      suffix:
        " according to his watch, and he is travelling down through the Scottish highlands to Inverness, tired and ever-so-slightly anxious in case he falls asleep between now and when the train reaches the station, and misses his cue to say to Alice, Drew and Aleesha: 'OK, this is Inverness, let's move it.'",
    },
    {
      time: "10.03",
      book: "Ctrl-Z",
      author: "Andrew Norriss",
      prefix:
        "The date was the 14th of May and the clock on his desk said the time was twenty three minutes past ten, so he tapped in the numbers 10.23. At least, that's what he meant to do. In fact he typed in the numbers ",
      suffix: ".",
    },
  ],
  "10:05": [
    {
      time: "five past ten",
      book: "The Undomestic Goddess",
      author: "Sophie Kinsella",
      prefix:
        "We both watch as a pair of swans sail regally under the little bridge. Then I glance at my watch. It's already ",
      suffix:
        ". “We should get going,” I say with a little start. Your mother will be waiting.” “There's no rush,” Nathaniel calls as I hasten down the other side of the bridge. “We've got all day.” He lopes down the bridge. “It's OK. You can slow down.” I try to match his relaxed pace. But I'm not used to this easy rhythm. I'm used to striding along crowded pavements, fighting my way, pushing and elbowing.",
    },
  ],
  "10:07": [
    {
      time: "10.07 am",
      book: "I Don't Know How She Does It",
      author: "Allison Pearson",
      prefix: "",
      suffix:
        ": In a meeting with Rod, Momo and Guy. We are rehearsing the final for the third time, with Rod and Guy taking the parts of the clients, when Rod's secretary, Lorraine, bursts in",
    },
  ],
  "10:09": [
    {
      time: "10:09",
      book: "American Tabloid",
      author: "James Ellroy",
      prefix: "He followed the squeals down a hallway. A wall clock read 8:09-",
      suffix: " Dallas time.",
    },
  ],
  "10:10": [
    {
      time: "10:10",
      book: "The Hollow Man",
      author: "John Dickson Carr",
      prefix: "",
      suffix: " Shot is fired",
    },
    {
      time: "ten minutes past 10",
      book: "England, Their England",
      author: "AG Macdonell",
      prefix: "Saturday morning was bright and sunny, and at ",
      suffix:
        " Donald arrived at the Embankment entrance of Charing Cross Underground Station, carrying a small suitcase full of clothes suitable for outdoor sports and pastimes.",
    },
  ],
  "10:11": [
    {
      time: "eleven minutes past ten",
      book: "Three Men and a Maid",
      author: "P.G. Wodehouse",
      prefix:
        "Through the curtained windows of the furnished apartment which Mrs. Horace Hignett had rented for her stay in New York rays of golden sunlight peeped in like the foremost spies of some advancing army. It was a fine summer morning. The hands of the Dutch clock in the hall pointed to thirteen minutes past nine; those of the ormolu clock in the sitting-room to ",
      suffix:
        "; those of the carriage clock on the bookshelf to fourteen minutes to six. In other words, it was exactly eight; and Mrs. Hignett acknowledged the fact by moving her head on the pillow, opening her eyes, and sitting up in bed. She always woke at eight precisely.",
    },
  ],
  "10:12": [
    {
      time: "Ten twelve",
      book: "The Undomestic Goddess",
      author: "Sophie Kinsella",
      prefix:
        "“I'll take the coffee tray out,” I suggest humbly. As I pick it up I glance again at my watch. ",
      suffix: ". I wonder if they've started the meeting.",
    },
    {
      time: "10:12 a.m.",
      book: "Freedom",
      author: "Jonathan Franzen",
      prefix:
        "He stood up once, early on, to lock his office door, and then he was reading the last page, and it was exactly ",
      suffix:
        ", and the sun beating on his office windows was a different sun from the one he'd always known.",
    },
  ],
  "10:13": [
    {
      time: "thirteen minutes past ten",
      book: "Michel Strogoff",
      author: "Jules Verne",
      prefix:
        '"By the bye," said the first, "I was able this morning to telegraph the very words of the order to my cousin at seventeen minutes past ten." "And I sent it to the Daily Telegraph at ',
      suffix: '." "Bravo, Mr. Blount!" "Very good, M. Jolivet."',
    },
  ],
  "10:14": [
    {
      time: "Ten fourteen",
      book: "Coyote Blue",
      author: "Christopher Moore",
      prefix: "“Okay. ",
      suffix:
        ": Mrs. Narada reports that her cat has been attacked by a large dog. Now I send all the boys out looking, but they don't find anything until eleven. Then one of them calls in that a big dog has just bitten holes in the tires on his golf cart and run off. Eleven thirty: Dr. Epstein makes his first lost-nap call: dog howling. Eleven thirty-five: Mrs. Norcross is putting the kids out on the deck for some burgers when a big dog jumps over the rail, eats the burgers, growls at the kids, runs off. First mention of lawsuit.”",
    },
  ],
  "10:15": [
    {
      time: "10.15",
      book: "Evil under the sun",
      author: "Agatha Christie",
      prefix: "At ",
      suffix:
        " Arlena departed from her rondezvous, a minute or two later Patrick Redfern came down and registered surprise, annoyance, etc. Christine's task was easy enough. Keeping her own watch concealed she asked she asked Linda at twenty-five past eleven what time it was. Linda looked at her watch and replied that it was a quarter to twelve.",
    },
  ],
  "10:16": [
    {
      time: "10:16",
      book: "The Undomestic Goddess",
      author: "Sophie Kinsella",
      prefix: "",
      suffix:
        " At last. Forty minutes of hard work and I have made precisely one bed. I'm way behind. But never mind. Just keep moving. Laundry next",
    },
  ],
  "10:17": [
    {
      time: "seventeen minutes past ten",
      book: "Michel Strogoff",
      author: "Jules Verne",
      prefix:
        '"By the bye," said the first, "I was able this morning to telegraph the very words of the order to my cousin at ',
      suffix:
        '." "And I sent it to the Daily Telegraph at thirteen minutes past ten."n "Bravo, Mr. Blount!" "Very good, M. Jolivet." "I will try and match that!"',
    },
  ],
  "10:18": [
    {
      time: "10:18",
      book: "Extremely Loud and Incredibly Close",
      author: "Jonathan Safran Foer",
      prefix: "I know that it was ",
      suffix: " when I got home because I look at my watch a lot.",
    },
  ],
  "10:20": [
    {
      time: "twenty past ten",
      book: "A Portrait of the Artist as a Young Man",
      author: "James Joyce",
      prefix:
        "How much is the clock fast now? His mother straightened the battered alarm clock that was lying on its side in the middle of the mantelpiece until its dial showed a quarter to twelve and then laid it once more on its side. An hour and twenty-five minutes, she said. The right time now is ",
      suffix: ".",
    },
  ],
  "10:21": [
    {
      time: "10.21",
      book: "The Radiant Way",
      author: "Margaret Drabble",
      prefix:
        "Liz Headleand stares into the mirror, as though entranced. She does not see herself or the objects on her dressing-table. The clock abruptly jerks to ",
      suffix: ".",
    },
  ],
  "10:22": [
    {
      time: "10:22",
      book: "Extremely Loud and Incredibly Close",
      author: "Jonathan Safran Foer",
      prefix:
        "I listened to them, and listened to them again, and then before I had time to figure out what to do, or even what to think or feel, the phone started ringing. It was ",
      suffix: ":27. I looked at the caller ID and saw that it was him.",
    },
  ],
  "10:23": [
    {
      time: "twenty three minutes past ten",
      book: "Ctrl-Z",
      author: "Andrew Norriss",
      prefix:
        "The date was the 14th of May and the clock on his desk said the time was ",
      suffix:
        ", so he tapped in the numbers 10.23. At least, that's what he meant to do. In fact he typed in the numbers 10.03",
    },
  ],
  "10:25": [
    {
      time: "10:25",
      book: "The Lost Honour of Katharina Blum",
      author: "Heinrich Böll",
      prefix: "",
      suffix:
        ": Phone call from Lüding, very worked up, urging me to return at once and get in touch with Alois, who was equally worked up",
    },
    {
      time: "10:25",
      book: "The Voices of Time",
      author: "JG Ballard",
      prefix:
        "One meal is enough now, topped up with a glucose shot. Sleep is still 'black', completely unrefreshing. Last night I took a 16 mm. film of the first three hours, screened it this morning at the lab. The first true-horror movie. I looked like a half-animated corpse. Woke ",
      suffix: ". To sleep 3:45.",
    },
  ],
  "10:26": [
    {
      time: "10:26",
      book: "The Undomestic Goddess",
      author: "Sophie Kinsella",
      prefix: "",
      suffix:
        " No. Please, no. I can hardly bear to look. It's a total disaster. Everything in the washing machine has gone pink. Every single thing. What happened? With trembling fingers I pick out a damp cashmere cardigan. It was a cream when I put it in. It's now a sickly shade of candy floss. I knew K3 was bad news. I knew it ",
    },
    {
      time: "ten-twenty-six",
      book: "Hard-boiled Wonderland and The End of the World",
      author: "Haruki Murakami",
      prefix:
        "In the exact centre of my visual field was the alarm clock, hands pointing to ",
      suffix: ". An alarm clock I received as a memento of somebody's wedding.",
    },
  ],
  "10:27": [
    {
      time: "twenty-seven minutes past 10",
      book: "England, Their England",
      author: "AG Macdonell",
      prefix: "Mr. Harcourt woke up with mysterious suddenness at ",
      suffix:
        ", and, by a curious coincidence, it was at that very instant that the butler came in with two footmen laden with trays of whisky, brandy, syphons, glasses, and biscuits.",
    },
    {
      time: "10:27 a.m.",
      book: "The Accidental",
      author: "Ali Smith",
      prefix: "She is on holiday in Norfolk. The substandard clock radio says ",
      suffix:
        " The noise is Katrina the Cleaner thumping the hoover against the skirting boards and the bedroom doors. Her hand is asleep. It is still hooked through the handstrap of the camera. She unhooks it and shakes it to get the blood back into it. She puts her feet on top of her trainers and slides them across the substandard carpet. It has the bare naked feet of who knows how many hundreds of dead or old people on it.",
    },
  ],
  "10:30": [
    {
      time: "10.30 a.m.",
      book: "The Curious Incident of the Dog in the Night-time",
      author: "Mark Haddon",
      prefix: "",
      suffix: " Brea",
    },
    {
      time: "ten thirty",
      book: "The Hours",
      author: "Michael Cunningham",
      prefix: "according to the clock on the wall, it is barely ",
      suffix: ".",
    },
    {
      time: "ten-thirty",
      book: "The Sportswriter",
      author: "Richard Ford",
      prefix: "At ",
      suffix:
        " I'm cleaned up, shaved and dressed in my Easter best - a two-piece seersucker Palm Beach I've had since college.",
    },
  ],
  "10:31": [
    {
      time: "Just after half past ten.",
      book: "Death on the Nile",
      author: "Agatha Christie",
      prefix: '"If you please. You went to bed at what time, Madame?" "',
      suffix: '"',
    },
  ],
  "10:32": [
    {
      time: "Just after half past ten.",
      book: "Death on the Nile",
      author: "Agatha Christie",
      prefix: '"If you please. You went to bed at what time, Madame?" "',
      suffix: '"',
    },
  ],
  "10:35": [
    {
      time: "Five-and-twenty to eleven",
      book: "Rope",
      author: "Patrick Hamilton",
      prefix: "",
      suffix:
        ". A horrible hour - a macabre hour, for it is not only the hour of pleasure ended, it is the hour when pleasure itself has been found wanting",
    },
  ],
  "10:36": [
    {
      time: "ten thirty-six",
      book: "The Adventure of the Missing Three-Quarter",
      author: "Arthur Conan Doyle",
      prefix: '"Strand post mark and dispatched ',
      suffix:
        '" said Holmes reading it over and over. "Mr Overton was evidently considerably excited when he sent it over and somewhat incoherent in consequence."',
    },
  ],
  "10:37": [
    {
      time: "10.37 a.m.",
      book: "The Unpleasantness at the Bellona Club",
      author: "Dorothy L. Sayers",
      prefix:
        "I quite agree with you,' said Mr Murbles. 'It is a most awkward situation. Lady Dormer died at precisely ",
      suffix: " on November 11th.'",
    },
  ],
  "10:38": [
    {
      time: "10:38",
      book: "The Undomestic Goddess",
      author: "Sophie Kinsella",
      prefix:
        "There must be a solution, there must be. Frantically I scan the cans of products stacked on the shelves. Stain Away. Vanish. There has to be a remedy. ... I just need to think. ... ",
      suffix:
        " OK, I have the answer. It may not totally work—but it's my best shot.",
    },
  ],
  "10:40": [
    {
      time: "10:40",
      book: "The Lost Honour of Katharina Blum",
      author: "Heinrich Böll",
      prefix: "",
      suffix:
        ": Call from Katharina asking me whether I had really said what was in the News",
    },
  ],
  "10:43": [
    {
      time: "10.43 a.m",
      book: "The Wish List",
      author: "Jane Costello",
      prefix: "24 January, ",
      suffix:
        ": one month and two days later I wonder if I should worry about the fact that my darling boyfriend bought me a birthday present that has the potential to cause instant death...",
    },
  ],
  "10:45": [
    {
      time: "quarter to eleven",
      book: "The Valley of Fear",
      author: "Arthur Conan Doyle",
      prefix:
        "If this is so, we have now to determine what Barker and Mrs. Douglas, presuming they are not the actual murderers, would have been doing from ",
      suffix:
        ", when the sound of the shot brought them down, until quarter past eleven, when they rang for the bell and summoned the servants'.",
    },
    {
      time: "quarter to eleven",
      book: "Harry Potter and the Chamber of Secrets",
      author: "J.K.Rowling",
      prefix: "They reached King's Cross at a ",
      suffix:
        ". Mr Weasley dashed across the road to get trolleys for their trunks and they all hurried into the station.",
    },
  ],
  "10:47": [
    {
      time: "10.47",
      book: "Trumpet",
      author: "Jackie Kay",
      prefix: "He whistles in the shower. It is ",
      suffix: " and he is ready for the off.",
    },
  ],
  "10:48": [
    {
      time: "10.48am",
      book: "Apple Tree Yard",
      author: "Louise Doughty",
      prefix: "At ",
      suffix:
        ", I closed my folder but didn't bother putting it back in my bag, so you knew I was on my way to a committee or meeting room nearby. Before I stood up, I folded my paper napkin and put it and the spoon into my coffee cup, a neat sort of person, you thought.",
    },
  ],
  "10:49": [
    {
      time: "forty-nine minutes past ten",
      book: "Narrative of a Journey round the Dead Sea and in the Bible lands in 1850 and 1851",
      author: "Félicien de Saulcy",
      prefix: "By ",
      suffix:
        ", we fall in again with a fine portion of the ancient road, which the modern track constantly follows, and descend by some steep windings, hewn in the side of a precipitous cliff, to the place where the Ouad-el-Haoud commences.",
    },
  ],
  "10:50": [
    {
      time: "10.50 a.m.",
      book: "The Curious Incident of the Dog in the Night-time",
      author: "Mark Haddon",
      prefix: "",
      suffix: " Art class with Mrs Peter",
    },
    {
      time: "ten to eleven",
      book: "Bomber",
      author: "Len Deighton",
      prefix:
        "As he walked back to the flight office, airmen were forming a line to await the arrival of the NAAFI van with morning tea and cakes. Lambert looked at his watch; it was ",
      suffix: ".",
    },
  ],
  "10:53": [
    {
      time: "10.53 hrs",
      book: "Spies",
      author: "Michael Frayn",
      prefix: "He begins to make a record of our observations.'",
      suffix:
        ",' he writes, as we crouch at the top of the stairs, listening to his mother in the hall below.",
    },
    {
      time: "10:53",
      book: "Austerlitz",
      author: "W.G. Sebald",
      prefix:
        "I gaze and gaze again at that face, which seems to me both strange and familiar, said Austerlitz, I run the tape back repeatedly, looking at the time indicator in the top left-hand corner of the screen, where the figures covering part of her forehead show the minutes and seconds, from ",
      suffix:
        " to 10:57, while the hundredths of a second flash by so fast that you cannot read and capture them.",
    },
  ],
  "10:55": [
    {
      time: "five minutes to eleven",
      book: "The House at Pooh Corner",
      author: "AA Milne",
      prefix: "The clock was still saying ",
      suffix: " when Pooh and Piglet set out on their way half an hour later.",
    },
  ],
  "10:57": [
    {
      time: "10.57",
      book: "Austerlitz",
      author: "W. G. Sebald",
      prefix:
        "I run the tape back repeatedly, looking at the time indicator in the top left-hand corner of the screen, where the figures covering part of her forehead show the minutes and seconds, from 10.53 to ",
      suffix: ".",
    },
  ],
  "10:58": [
    {
      time: "10:58",
      book: "Lightning Rods",
      author: "Helen DeWitt",
      prefix:
        "One day Joe was sitting in the office waiting for his 11 o'clock appointment, and at ",
      suffix: " this black gal came in.",
    },
  ],
  "10:59": [
    {
      time: "one minute to eleven",
      book: "Harry Potter and the Half-Blood Prince",
      author: "J. K. Rowling",
      prefix:
        "Harry grunted in his sleep and his face slid down the window an inch or so, making his glasses still more lopsided, but he did not wake up. An alarm clock, repaired by Harry several years ago, ticked loudly on the sill, showing ",
      suffix: ".",
    },
  ],
  "11:00": [
    {
      time: "eleven o'clock",
      book: "Mrs Dalloway",
      author: "Virginia Woolf",
      prefix:
        "'Who can - what can -” asked Mrs Dalloway (thinking it was outrageous to be interrupted at ",
      suffix:
        " on the morning of the day she was giving a party), hearing a step on the stairs.",
    },
    {
      time: "11 o'clock",
      book: "The End of Mr Y",
      author: "Scarlett Thomas",
      prefix: '"By ',
      suffix:
        ' I have finished the first chapter of Mr Y. The winter sun is peeping meekly through the thin curtains and I decide to get up"',
    },
    {
      time: "at eleven",
      book: "The Weather in the Streets",
      author: "Rosamond Lehmann",
      prefix:
        "He was rather a long time, and I began to feel muffled, weighed down by thick stuffs and silence. I thought: He'll never come back; and when he did his figure seemed to come at me from very far away, dream-like and dwindled, making his way back along a tunnel...I dare say it was champagne ",
      suffix: " in the morning.",
    },
    {
      time: "11 o'clock",
      book: "Scoop",
      author: "Evelyn Waugh",
      prefix:
        "As her husband had told him, she was still in bed although it was past ",
      suffix:
        ". Her normally mobile face was encased in clay, rigid and menacing as an Aztec mask.",
    },
    {
      time: "eleven",
      book: "Mrs. Dalloway",
      author: "Virginia Woolf",
      prefix:
        "As they looked the whole world became perfectly silent, and a flight of gulls crossed the sky, first one gull leading, then another, and in this extraordinary silence and peace, in this pallor, in this purity, bells struck ",
      suffix: " times the sound fading up there among the gulls.",
    },
    {
      time: "eleven o'clock",
      book: "The Snowman",
      author: "Jo Nesbo",
      prefix: "At ",
      suffix:
        " in the morning, large flakes had appeared from a colourless sky and invaded the fields, gardens and lawns of Romerike like an armada from outer space.",
    },
    {
      time: "eleven o'clock",
      book: "The Long Dark Tea-Time of the Soul",
      author: "Douglas Adams",
      prefix: "At ",
      suffix:
        " the phone rang, and still the figure did not respond, any more than it had responded when the phone had rung at twenty-five to seven, and again for ten minutes continuously starting at five to seven...",
    },
    {
      time: "eleven o'clock",
      book: "Mrs Dalloway",
      author: "Virginia Woolf",
      prefix:
        "Big Ben was striking as she stepped out into the street. It was ",
      suffix:
        " and the unused hour was fresh as if issued to children on a beach.",
    },
    {
      time: "eleven o'clock",
      book: "The big sleep",
      author: "Raymond Chandler",
      prefix: "It was about ",
      suffix:
        " in the morning, mid October, with the sun not shining and a look of hard wet rain in the clearness of the foothills. I was wearing my powder-blue suit, with dark blue shirt, tie and display handkerchief, black brogues, black wool socks with dark blue clocks on them. I was neat, clean, shaved and sober, and I didn't care who knew it. I was everything the well-dressed private detective ought to be. I was calling on four millon dollars.",
    },
    {
      time: "eleven",
      book: "The Saints",
      author: "Patsy Hickman",
      prefix:
        "My sister is terrified that I might write and tell all the family secrets. Why do I feel like a rebel, like an iconoclast? I am only trying to do a writing class, what is wrong with that? I keep telling myself that once in the car I will be fine, I can listen to Radio Four Woman’s Hour and that will take me till ",
      suffix: " o’clock when the class starts.",
    },
    {
      time: "eleven o'clock",
      book: "The Mysteries of London",
      author: "G.W.M. Reynolds",
      prefix:
        "ON the morning following the events just narrated, Mrs. Arlington was seated at breakfast in a sweet little parlour of the splendid mansion which the Earl of Warrington had taken and fitted up for her in Dover Street, Piccadilly. It was about ",
      suffix:
        "; and the Enchantress was attired in a delicious deshabillé. With her little feet upon an ottoman near the fender, and her fine form reclining in a luxurious large arm-chair, she divided her attention between her chocolate and the columns of the Morning Herald. She invariably prolonged the morning's repast as much as possible, limply because it served to wile away the time until the hour for dressing arrived.",
    },
    {
      time: "Eleven o'Clock",
      book: "Twice Around the Clock",
      author: "George Augustus Sala",
      prefix: "Quiet as I am, I become at ",
      suffix:
        " in the Morning on every day of the week save Sunday a raving, ranting maniac -- a dangerous lunatic, panting with insane desires to do, not only myself but other people, a mischief, and possessed less by hallucination than by rabies.",
    },
    {
      time: "eleven",
      book: "The Stranger's Child",
      author: "Alan Hollinghurst",
      prefix:
        "Though perhaps' – but here the bracket clock whirred and then hectically struck ",
      suffix:
        ", its weights spooling downwards at the sudden expense of energy. She had to sit for a moment, when the echo had vanished, to repossess her thoughts.",
    },
    {
      time: "at eleven",
      book: "Three Men in a Boat",
      author: "Jerome K Jerome",
      prefix: "We got to Waterloo ",
      suffix:
        ", and asked where the eleven-five started from. Of course nobody knew; nobody at Waterloo ever does know where a train is going to start from, or where a train when it does start is going to, or anything about it.",
    },
    {
      time: "at eleven",
      book: "Three Men in a Boat",
      author: "Jerome K Jerome",
      prefix: "We got to Waterloo ",
      suffix:
        ", and asked where the eleven-five started from.Of course nobody knew; nobody at Waterloo ever does know where a train is going to start from, or where a train when it does start is going to, or anything about it.",
    },
    {
      time: "eleven o'clock",
      book: "Frankenstein",
      author: "Mary Shelley",
      prefix: "We passed a few sad hours until ",
      suffix:
        ", when the trial was to commence. My father and the rest of the family being obliged to attend as witnesses, I accompanied them to the court. During the whole of this wretched mockery of justice I suffered living torture.",
    },
  ],
  "11:01": [
    {
      time: "just past eleven",
      book: "Mary and O'Neil",
      author: "Justin Cronin",
      prefix:
        "O'Neil rises and takes the tray. He has finished the tea, but the muffins are still here in a wicker basket covered with a blue napkin. The clock above the stove says that it is ",
      suffix: ", and guests will be arriving at the house now.",
    },
  ],
  "11:02": [
    {
      time: "11.02am",
      book: "The Atomic Bombings of Hiroshima and Nagasaki",
      author: "The Manhattan Engineer District",
      prefix: "On August 9th, three days later, at ",
      suffix:
        ", another B−29 dropped the second bomb on the industrial section of the city of Nagasaki, totally destroying 1 1/2 square miles of the city, killing 39,000 persons and injuring 25,000 more.",
    },
  ],
  "11:03": [
    {
      time: "Eleven oh-three",
      book: "Little Green Men",
      author: "Christopher Buckley",
      prefix:
        '"What makes you think it\'s for real?" "Just a hunch, really. He sounded for real. Sometimes you can just tell about people"-he smiled-"even if you\'re a dull old WASP." "I think it\'s a setup." "Why?" "I just do. Why would someone from the government want to help you?" "Good question. Guess I\'ll find out." She went back into the kitchen."What time are you meeting him?" she called out. "',
      suffix:
        '," he said. "That made me think he\'s for real. Military and intelligence types set precise appointment times to eliminate confusion and ambiguity. Nothing ambiguous about eleven oh-three."',
    },
    {
      time: "11.03am",
      book: "Death and the Compass",
      author: "Jorge Luis Borges",
      prefix: "On the fourth, at ",
      suffix:
        ", the editor of the Yidische Zaitung put in a call to him; Doctor Yarmolinsky did not answer. He was found in his room, his face already a little dark, nearly nude beneath a large, anachronistic cape.",
    },
  ],
  "11:04": [
    {
      time: "past 11 o'clock",
      book: "Scoop",
      author: "Evelyn Waugh",
      prefix:
        "As her husband had told him, she was still in bed although it was ",
      suffix:
        ". Her normally mobile face was encased in clay, rigid and menacing as an Aztec mask.",
    },
  ],
  "11:05": [
    {
      time: "11:05",
      book: "The Voices of Time",
      author: "JG Ballard",
      prefix:
        "July 3: 5 3/4 hours. Little done today. Deepening lethargy, dragged myself over to the lab, nearly left the road twice. Concentrated enough to feed the zoo and get the log up to date. Read through the operating manuals Whitby left for the last time, decided on a delivery rate of 40 rontgens/min., target distance of 530 cm. Everything is ready now. Woke ",
      suffix: ". To sleep 3:15.",
    },
    {
      time: "five past eleven",
      book: "Gone Tomorrow",
      author: "Lee Child",
      prefix: "Sansom arrived in a Town Car at ",
      suffix:
        ". Local plates, which meant he had ridden up most of the way on the train. Less convenient for him, but a smaller carbon footprint than driving all the way, or flying. Every detail mattered, in a campaign.",
    },
  ],
  "11:06": [
    {
      time: "11:06",
      book: "The Undomestic Goddess",
      author: "Sophie Kinsella",
      prefix: "",
      suffix: " And ... oh. The ironing. What am I going to do about that",
    },
    {
      time: "11.06am",
      book: "One of Our Thursdays is Missing",
      author: "Jasper Fforde",
      prefix:
        "Despite the remaking of the BookWorld, some books remained tantalisingly out of reach [...] It was entirely possible that they didn't know there was a BookWorld, and still they thought they were real. A fantastic notion, until you consider that up until ",
      suffix: " on 12 April 1948, everyone else had thought the same.",
    },
  ],
  "11:07": [
    {
      time: "seven minutes past eleven",
      book: "The Adventure Club Afloat",
      author: "Ralph Henry Barbour",
      prefix: "At exactly ",
      suffix:
        " by the ship's clock the Adventurer gave a prolonged screech and, moorings cast off, edged her way out of the basin and dipped her nose in the laughing waters of the bay, embarked at last on a voyage that was destined to fully vindicate her new name.",
    },
  ],
  "11:08": [
    {
      time: "eight minutes past eleven",
      book: "Stephen Hero",
      author: "James Joyce",
      prefix:
        "The bursar was standing in the hall with his arms folded across his chest and when he caught sight of the fat young man he looked significantly at the clock. It was ",
      suffix: ".",
    },
  ],
  "11:09": [
    {
      time: "around eleven",
      book: "Where I'm Calling From",
      author: "Raymond Carver",
      prefix: "The first time I saw them it was ",
      suffix:
        ", eleven-fifteen, a Saturday morning, I was about two thirds through my route when I turned onto their block and noticed a '56 Ford sedan pulled up in the yard with a big open U-Haul behind.",
    },
  ],
  "11:10": [
    {
      time: "Ten minutes after eleven",
      book: "Emotionally Weird",
      author: "Kate Atkinson",
      prefix: "",
      suffix:
        " in Archie McCue's room on the third floor of the extension to the Robert Matthews' soaring sixties' tower - The Queen's Tower, although no queen was ever likely to live in it",
    },
  ],
  "11:12": [
    {
      time: "11:12",
      book: "The Undomestic Goddess",
      author: "Sophie Kinsella",
      prefix: "",
      suffix:
        " I have a solution, via the local paper. A girl from the village will collect it, iron it all overnight at £3 a shirt, and sew on Eddie's button",
    },
    {
      time: "11:12",
      book: "A Thousand Acres",
      author: "Jane Smiley",
      prefix: "I squinted down the street at the bank clock: ",
      suffix:
        ", 87 degrees. \"It's only a block and a half and it's not that hot, Daddy. The walk will do you good.\" This conversation made me breathless, as if I were wearing a girdle with tight stays.",
    },
  ],
  "11:14": [
    {
      time: "11.14am.",
      book: "The Girl With The Dragon Tattoo",
      author: "Stieg Larsson",
      prefix: "The report was dated Sunday, 25 September, 1966, at ",
      suffix:
        " The text was laconic. Call from Hrk Vanger; stating that his brother's daughter (?) Harriett Ulrika Vanger, born 15 June 1960 (age 1960) has been missing from her home on Hedley Island since Saturday afternoon.",
    },
  ],
  "11:15": [
    {
      time: "11:15",
      book: "The Adventures of Sherlock Holmes",
      author: "Sir Arthur Conan Doyle",
      prefix:
        '"Have you a couple of days to spare? Have just been wired for from the west of England in connection with Boscombe Valley tragedy. Shall be glad if you will come with me. Air and scenery perfect. Leave Paddington by the ',
      suffix: '."',
    },
    {
      time: "eleven-fifteen",
      book: "Where I'm Calling From",
      author: "Raymond Carver",
      prefix: "The first time I saw them it was around eleven, ",
      suffix:
        ", a Saturday morning, I was about two thirds through my route when I turned onto their block and noticed a '56 Ford sedan pulled up in the yard with a big open U-Haul behind. There are only three houses on Pine, and theirs was the last house,the others being the Murchisons, who'd been in Arcata a little less than a year, and the Grants, who'd been here about two years. Murchison worked at Simpson Redwood, and Gene Grant was a cook on the morning shift at Denny's. Those two, then a vacant lot, then the house on the end that used to belong to the Coles.",
    },
  ],
  "11:17": [
    {
      time: "seventeen minutes past eleven",
      book: "Dubliners",
      author: "James Joyce",
      prefix:
        "Mrs. Mooney glanced instinctively at the little gilt clock on the mantelpiece as soon as she had become aware through her revery that the bells of George's Church had stopped ringing. It was ",
      suffix:
        ": she would have lots of time to have the matter out with Mr. Doran and then catch short twelve at Marlborough Street. She was sure she would win.",
    },
  ],
  "11:18": [
    {
      time: "11.18",
      book: "Trumpet",
      author: "Jackie Kay",
      prefix: "It is ",
      suffix:
        ". A row of bungalows in a round with a clump of larch tree in the middle.",
    },
  ],
  "11:19": [
    {
      time: "11:19",
      book: "Blackout",
      author: "Connie Willis",
      prefix:
        "A whistle cut sharply across his words. Peter got onto his knees to look out the window, and Miss Fuller glared at him. Polly looked down at her watch: ",
      suffix: ". The train. But the stationmaster had said it was always late.",
    },
  ],
  "11:20": [
    {
      time: "11h20",
      book: "Moxyland",
      author: "Lauren Beukes",
      prefix: "OFFICER'S NOTES Disruption alert logged ",
      suffix:
        " from Stones' Pool Hall (Premises ID 33CBD-Long181). Officer and Aito /379 responded. On arrival found subject shouting threats and acting in aggressive manner. A scan of the subject's SIM ID register revealed that the subject has recent priors including previous public disruptions and a juvenile record.",
    },
    {
      time: "11.20",
      book: "American Gods",
      author: "Neil Gaiman",
      prefix:
        "Sweeney pointed to the clock above the bar, held in the massive and indifferent jaws of a stuffed alligator head. The time was ",
      suffix: ".",
    },
  ],
  "11:25": [
    {
      time: "twenty-five past eleven",
      book: "Evil under the Sun",
      author: "Agatha Christie",
      prefix:
        "At 10.15 Arlena departed from her rondezvous, a minute or two later Patrick Redfern came down and registered surprise, annoyance, etc. Christine's task was easy enough. Keeping her own watch concealed she asked Linda at ",
      suffix:
        " what time it was. Linda looked at her watch and replied that it was a quarter to twelve.",
    },
    {
      time: "11.25am",
      book: "The Lost Honour of Katharina Blum",
      author: "Heinrich Böll",
      prefix: "When, at about ",
      suffix:
        ", Katharina Blum was finally taken from her apartment for questioning, it was decided not to handcuff her at all.",
    },
  ],
  "11:27": [
    {
      time: "11.27",
      book: "The Second Internet Cafe, Part 2: The Cascade Annihilator",
      author: "Chris James",
      prefix: "It's from one of the more recent plates the tree has scanned: ",
      suffix: " in the morning of 4 April 1175",
    },
  ],
  "11:28": [
    {
      time: "twenty-eight minutes past eleven",
      book: "Sinister Street",
      author: "Compton Mackenzie",
      prefix:
        "From twenty minutes past nine until twenty-seven minutes past nine, from twenty-five minutes past eleven until ",
      suffix:
        ", from ten minutes to three until two minutes to three the heroes of the school met in a large familiarity whose Olympian laughter awed the fearful small boy that flitted uneasily past and chilled the slouching senior that rashly paused to examine the notices in assertion of an unearned right.",
    },
  ],
  "11:29": [
    {
      time: "twenty-nine minutes after eleven, a.m.",
      book: "Around the World in Eighty Days",
      author: "Jules Verne",
      prefix:
        "You are four minutes too slow. No matter; it's enough to mention the error. Now from this moment, ",
      suffix: ", this Wednesday, 2nd October, you are in my service.",
    },
  ],
  "11:30": [
    {
      time: "11.30",
      book: "Singularity Sky",
      author: "Charles Stross",
      prefix: "'It is now ",
      suffix:
        ". The door to this room is shut, and will remain shut, barring emergencies, until 12.00. I am authorised to inform you that we are now under battle orders.",
    },
    {
      time: "half-past eleven",
      book: "Far from the madding crowd",
      author: "Thomas Hardy",
      prefix:
        "\"O, Frank - I made a mistake! - I thought that church with the spire was All Saints', and I was at the door at ",
      suffix: ' to a minute as you said..."',
    },
    {
      time: "half-past eleven",
      book: "To the Devil a Daughter",
      author: "Dennis Wheatley",
      prefix:
        '"Thank-you," said C.B. quietly; but as he hung up his face was grim. In a few minutes he would have to break it to John that, although they had braved such dredful perils dring the earlier part of the night they had, after all, failed to save Christina. Beddows had abjured Satan at a little after ',
      suffix:
        '. By about eighteen minutes the Canon had beaten them to it again."',
    },
    {
      time: "11.30",
      book: "The Wind-up Bird Chronicle",
      author: "Haruki Murakami",
      prefix: "This time it was Kumiko. The wall clock said ",
      suffix: ".",
    },
  ],
  "11:31": [
    {
      time: "1131",
      book: "The Hunt for Red October",
      author: "Tom Clancy",
      prefix: "Albatross 8 passed over Pamlico Sound at ",
      suffix:
        " local time. Its on-board programming was designed to trace thermal receptors over the entire visible horizon, interrogating everything in sight and locking on any signature that fit its acquisition parameters.",
    },
  ],
  "11:32": [
    {
      time: "eleven thirty two",
      book: "Finnegans Wake",
      author: "James Joyce",
      prefix:
        "And after that, not forgetting, there was the Flemish armada, all scattered, and all officially drowned, there and then, on a lovely morning, after the universal flood, at about ",
      suffix: " was it? Off the coast of Cominghome...",
    },
  ],
  "11:34": [
    {
      time: "11.34am",
      book: "How to Fare Well and Stay Fair",
      author: "Adnan Mahmutovic",
      prefix: "Christmas Eve 1995. ",
      suffix:
        '. The first time, Almasa says it slowly and softly, as if she is really looking for an answer, "Are you talking to me?" She peers into the small, grimy mirror in a train toilet.',
    },
  ],
  "11:35": [
    {
      time: "11.35",
      book: "Our Man in Havana",
      author: "Graham Greene",
      prefix: "At ",
      suffix:
        " the Colonel came out; he looked hot and angry as he strode towards the lift. There goes a hanging judge, thought Wormold.",
    },
  ],
  "11:36": [
    {
      time: "Eleven thirty-six",
      book: "Losing You",
      author: "Nicci French",
      prefix:
        "I ran up the stairs, away from the heat and the noise, the mess and the confusion. I saw the clock radio by my bed. ",
      suffix: ".",
    },
  ],
  "11:38": [
    {
      time: "11:38",
      book: "The Circle",
      author: "Dave Eggers",
      prefix: "At ",
      suffix:
        ", she left her desk and walked to the side door of the auditorium, arriving ten minutes before noon.",
    },
  ],
  "11:40": [
    {
      time: "11.40am",
      book: "Around the World in Eighty Days",
      author: "Jules Verne",
      prefix:
        'Did escape occur to him? … But the door was locked, and the window heavily barred with iron rods. He sat down again, and drew his journal from his pocket. On the line where these words were written, "21st December, Saturday, Liverpool," he added, "80th day, ',
      suffix: '," and waited.',
    },
    {
      time: "twenty minutes before noon",
      book: "The Master of Go",
      author: "Yusunari Kawabata",
      prefix:
        "During the sessions at Ito he read the Lotus Sutra on mornings of play, and he now seemed to be bringing himself to order through silent meditation. Then, quickly, there came a rap of stone on board. It was ",
      suffix: ".",
    },
  ],
  "11:41": [
    {
      time: "Eleven forty-one",
      book: "Coyote Blue",
      author: "Christopher Moore",
      prefix: 'Spagnola took a deep breath and started into the log again. "',
      suffix:
        ": large dog craps in Dr. Yamata's Aston Martin. Twelve oh-three: dog eats two, count 'em, two of Mrs. Wittingham's Siamese cats. She just lost her husband last week; this sort of put her over the edge. We had to call Dr. Yamata in off the putting green to give her a sedative. The personal-injury lawyer in the unit next to hers was home for lunch and he came over to help. He was talking class action then, and we didn't even know who owned the dog yet.\"",
    },
  ],
  "11:42": [
    {
      time: "11:42",
      book: "The Undomestic Goddess",
      author: "Sophie Kinsella",
      prefix: "",
      suffix:
        " I'm doing fine. I'm doing well. I've got the Hoover on, I'm cruising along nicely- What was that? What just went up the Hoover? Why is it making that grinding noise? Have I broken it",
    },
  ],
  "11:44": [
    {
      time: "quarter to twelve",
      book: "Evil under the Sun",
      author: "Agatha Christie",
      prefix:
        "At 10.15 Arlena departed from her rondezvous, a minute or two later Patrick Redfern came down and registered surprise, annoyance, etc. Christine's task was easy enough. Keeping her own watch concealed she asked Linda at twenty-five past eleven what time it was. Linda looked at her watch and replied that it was a ",
      suffix: ".",
    },
  ],
  "11:45": [
    {
      time: "quarter to twelve",
      book: "Far from the madding crowd",
      author: "Thomas Hardy",
      prefix: '"...I waited till a ',
      suffix:
        ", and found then that I was in All Souls'. But I wasn't much frightened, for I thought it could be tomorrow as well.\"",
    },
    {
      time: "quarter to twelve",
      book: "Mrs. Dalloway",
      author: "Virginia Woolf",
      prefix:
        '"I will tell you the time," said Septimus, very slowly, very drowsily, smiling mysteriously. As he sat smiling at the dead man in the grey suit the quarter struck, the ',
      suffix: ".",
    },
    {
      time: "quarter to twelve",
      book: "Mrs Dalloway",
      author: "Virginia Woolf",
      prefix: "As he sat smiling, the quarter struck - the ",
      suffix: ".",
    },
    {
      time: "11.45am",
      book: "Epitaph for a Spy",
      author: "Eric Ambler",
      prefix:
        "I arrived at St. Gatien from Nice on Tuesday, the 14th of August. I was arrested at ",
      suffix:
        " on Thursday, the 16th by an agent de police and an inspector in plain clothes and taken to the Commissariat.",
    },
    {
      time: "11:45 A.M.",
      book: "Mistaken Identity",
      author: "Lisa Scottoline",
      prefix:
        "She tucked the phone in the crook of her neck and thumbed hurriedly through her pink messages. .... Dr. Provetto, at ",
      suffix: "",
    },
  ],
  "11:46": [
    {
      time: "quarter to twelve",
      book: "Evil under the Sun",
      author: "Agatha Christie",
      prefix:
        "At 10.15 Arlena departed from her rondezvous, a minute or two later Patrick Redfern came down and registered surprise, annoyance, etc. Christine's task was easy enough. Keeping her own watch concealed she asked Linda at twenty-five past eleven what time it was. Linda looked at her watch and replied that it was a ",
      suffix: ".",
    },
  ],
  "11:47": [
    {
      time: "thirteen minutes to noon",
      book: "The Kindly Ones",
      author: "Jonathan Littell",
      prefix:
        "It was a vast plain with no one on it, neither living on the earth nor dead beneath it; and I walked a long time beneath a colourless sky, which didn't let me judge the time (my watch, set like all military watches to Berlin time, hadn't stood up to the swim and showed an eternal ",
      suffix: ").",
    },
  ],
  "11:48": [
    {
      time: "ten minutes before noon",
      book: "The Circle",
      author: "Dave Eggers",
      prefix:
        "At 11:38, she left her desk and walked to the side door of the auditorium, arriving ",
      suffix: ".",
    },
  ],
  "11:50": [
    {
      time: "ten minutes to twelve",
      book: "The Adventure of Johnnie Waverley: A Hercule Poirot Story",
      author: "Agatha Christie",
      prefix:
        "The man who gave them to him handed him a ten-shilling note and promised him another if it were delivered at exactly ",
      suffix: ".",
    },
  ],
  "11:51": [
    {
      time: "nine minutes to twelve",
      book: "Lanterns & Lances",
      author: "James Thurber",
      prefix: "The next day, at ",
      suffix:
        " o'clock noon, the last clock ran down and stopped. It was then placed in the town museum, as a collector's item, or museum piece, with proper ceremonies, addresses, and the like.",
    },
  ],
  "11:52": [
    {
      time: "eight minutes to twelve",
      book: "Black Beauty",
      author: "Anna Sewell",
      prefix:
        "At any rate, we whirled into the station with many more, just as the great clock pointed to ",
      suffix:
        ' o\'clock. "Thank God! We are in time," said the young man, "and thank you, too, my friend, and your good horse..."',
    },
  ],
  "11:54": [
    {
      time: "six minutes to twelve",
      book: "Hangover Square",
      author: "Patrick Hamilton",
      prefix:
        "He swilled off the remains of [his beer] and looked at the clock. It was ",
      suffix: ".",
    },
  ],
  "11:55": [
    {
      time: "five minutes to twelve",
      book: "The Ragged Trousered Philanthropists",
      author: "Robert Tressell",
      prefix: "He was tearing off on his bicycle to one of the jobs about ",
      suffix:
        " to see if he could catch anyone leaving off for dinner before the proper time.",
    },
    {
      time: "11:55 a.m.",
      book: "All the President's Men",
      author: "Bernstein & Woodward",
      prefix: "It was ",
      suffix: " on April 30.",
    },
    {
      time: "11:55",
      book: "Kafka on the Shore",
      author: "Haruki Murakami",
      prefix: "What time did you arrive at the site? It was ",
      suffix:
        ". I remember since I happened to glance at my watch when we got there. We rode our bicycles to the bottom of the hill, as far as we could go, then climbed the rest of the way on foot.",
    },
  ],
  "11:56": [
    {
      time: "around noon",
      book: "Odalisque: The Baroque Cycle #3",
      author: "Neal Stephenson",
      prefix: "A few minutes' light ",
      suffix:
        " is all that you need to discover the error, and re-set the clock – provide that you bother to go up and make the observation.",
    },
    {
      time: "can't be far-off twelve",
      book: "The Ragged Trousered Philanthropists",
      author: "Robert Tressell",
      prefix:
        "I wondered what the time is?' said the latter after a pause'. 'I don't know exactly', replied Easton, 'but it ",
      suffix: ".'",
    },
  ],
  "11:57": [
    {
      time: "can't be far-off twelve",
      book: "The Ragged Trousered Philanthropists",
      author: "Robert Tressell",
      prefix:
        "I wondered what the time is?' said the latter after a pause'. 'I don't know exactly', replied Easton, 'but it ",
      suffix: ".'",
    },
  ],
  "11:58": [
    {
      time: "11.58",
      book: "11/22/63",
      author: "Stephen King",
      prefix: "And when you go down the steps, it's always ",
      suffix: " on the morning of September ninth, 1958.",
    },
    {
      time: "Two minutes before the clock struck noon",
      book: "Burlesques",
      author: "William Makepeace Thackeray",
      prefix: "",
      suffix:
        ", the savage baron was on the platform to inspect the preparation for the frightful ceremony of mid-day. The block was laid forth-the hideous minister of vengeance, masked and in black, with the flaming glaive in his hand, was ready. The baron tried the edge of the blade with his finger, and asked the dreadful swordsman if his hand was sure? A nod was the reply of the man of blood. The weeping garrison and domestics shuddered and shrank from him. There was not one there but loved and pitied the gentle lady",
    },
  ],
  "11:59": [
    {
      time: "near to twelve",
      book: "The Adventure of Johnnie Waverley: A Hercule Poirot Story",
      author: "Agatha Christie",
      prefix: "There is a big grandfather clock there, and as the hands drew ",
      suffix: " I don't mind confessing I was as nervous as a cat.",
    },
  ],
  "12:00": [
    {
      time: "twelve",
      book: "Wuthering Heights",
      author: "Emily Brontë",
      prefix:
        "'There's nobody here!' I insisted. 'It was yourself, Mrs. Linton: you knew it a while since.' 'Myself!' she gasped, 'and the clock is striking ",
      suffix: "!",
    },
    {
      time: "twelve",
      book: "The Brothers Karamazov",
      author: "Fyodor Dostoyevsky",
      prefix: "A cheap little clock on the wall struck ",
      suffix: " hurriedly, and served to begin the conversation.",
    },
    {
      time: "noon",
      book: "The Woman Who Had Two Navels",
      author: "Nick Joaquin",
      prefix:
        "He had saved [the republic] and it was now in the present, alive now and everywhere in the present, and the hovering faces brightened and blurred about him, became the sound of a canal in the morning, the look of some roofs in the ",
      suffix:
        " sun, and the fragrance of a certain evening flower. Here he was, home at last. Behind him were the mountains and the Sleeping Woman in the sky, and before him, like smoky flames in the sunset, the whole beautiful beloved city.",
    },
    {
      time: "twelve o'clock",
      book: "Mrs Dalloway",
      author: "Virginia Woolf",
      prefix: "It was precisely ",
      suffix:
        "; twelve by Big Ben; whose stroke was wafted over the northern part of London; blent with that of other clocks, mixed in a thin ethereal way with the clouds and wisps of smoke and died up there among the seagulls, twelve o'clock struck as Clarissa Dalloway laid her green dress on her bed and the Warren Smiths walked down Harley Street. Twelve was the hour of their appointment.",
    },
    {
      time: "twelve o'clock",
      book: "Mrs Dalloway",
      author: "Virginia Woolf",
      prefix: "It was precisely ",
      suffix:
        "; twelve by Big Ben; whose stroke was wafted over the northern part of London; blent with that of other clocks, mixed in a thin ethereal way wth the clouds and wisps of smoke and died up there among the seagulls - twelve o'clock struck as Clarissa Dalloway laid her green dress on the bed, and the Warren Smiths walked down Harley Street.",
    },
    {
      time: "twelve o’clock",
      book: "Mrs Dalloway",
      author: "Virginia Woolf",
      prefix: "It was precisely ",
      suffix:
        "; twelve by Big Ben; whose stroke was wafted over the northern part of London; blent with that of other clocks, mixed in a thin ethereal way with the clouds and wisps of smoke and died up there among the seagulls—twelve o’clock struck as Clarissa Dalloway laid her green dress on her bed, and the Warren Smiths walked down Harley Street",
    },
    {
      time: "twelve o’clock",
      book: "Mrs Dalloway",
      author: "Virginia Woolf",
      prefix: "It was precisely ",
      suffix:
        "; twelve by Big Ben; whose stroke was wafted over the northern part of London; blent with that of other clocks, mixed in a thin ethereal way with the clouds and wisps of smoke, and died up there among the seagulls.",
    },
    {
      time: "Noon",
      book: "tinkers",
      author: "Paul Harding",
      prefix: "",
      suffix:
        " found him momentarily alone, while the family prepared lunch in the kitchen. The cracks in the ceiling widened into gaps. The locked wheels of his bed sank into new fault lines opening in the oak floor beneath the rug. At any moment the floor was going to give",
    },
    {
      time: "noon",
      book: "The Bridge of San Luis Rey",
      author: "Thornton Wilder",
      prefix: "On Friday ",
      suffix:
        ", July the twentieth, 1714, the finest bridge in all Peru broke and precipitated five travellers into the gulf below.",
    },
    {
      time: "noon",
      book: "The Great Gatsby",
      author: "F. Scott Fitzgerald",
      prefix: "Roaring ",
      suffix:
        ". In a well-fanned Forty-second Street cellar I met Gatsby for lunch.",
    },
    {
      time: "Noon",
      book: "The Birds begun at Four o'clock",
      author: "Emily Dickinson",
      prefix:
        "The Birds begun at Four o'clock — Their period for Dawn — A Music numerous as space — But neighboring as ",
      suffix: " —",
    },
    {
      time: "twelve",
      book: "The Oxen",
      author: "Thomas Hardy",
      prefix: "The Oxen Christmas Eve, and ",
      suffix:
        ' of the clock. "Now they are all on their knees," An elder said as we sat in a flock By the embers in hearthside ease. We pictured the meek mild creatures where They dwelt in their strawy pen, Nor did it occur to one of us there To doubt they were kneeling then. So fair a fancy few would weave In these years! Yet, I feel, If someone said on Christmas Eve, "Come; see the oxen kneel "In the lonely barton by yonder coomb Our childhood used to know," I should go with him in the gloom, Hoping it might be so.',
    },
    {
      time: "noon",
      book: "The Tanners",
      author: "Robert Walser",
      prefix: "Then came the stroke of ",
      suffix:
        ", and all these working and professional people dispersed like a trampled anthill into all the streets and directions. The white bridge was swarming with nimble dots. And when you considered that each dot had a mouth with which it was now planning to eat lunch, you couldn't help bursting into laughter.",
    },
  ],
  "12:01": [
    {
      time: "12:01",
      book: "Chronopolis",
      author: "J.G. Ballard",
      prefix:
        "And on all sides there were the clocks. Conrad noticed them immediately, at every street corner, over every archway, three quarters of the way up the sides of buildings, covering every conceivable angle of approach. Most of them were too high off the ground to be reached by anything less than a fireman's ladder and still retained their hands. All registered the same time: ",
      suffix:
        ". Conrad looked at his wristwatch, noted that it was just 2:45. ‘‘They were driven by a master dock’’ Stacey told him. ‘‘When that stopped, they all ceased at the same moment. One minute after midnight, thirty-seven years ago.’’",
    },
    {
      time: "12:01",
      book: "Boy A",
      author: "Jonathan Trigell",
      prefix:
        "It was the twelfth of December, the twelfth month. A was twelve. The electric clock/radio by his bedside table said ",
      suffix: ".",
    },
    {
      time: "12:01",
      book: "Boy A",
      author: "Jonathan Trigell",
      prefix:
        "It was the twelfth of December, the twelfth month. A was twelve. The electric clock/radio by his bedside table said ",
      suffix:
        ". A was waiting for it to read 12:12, he hoped there would be some sense of cosmic rightness when it did.",
    },
  ],
  "12:02": [
    {
      time: "twelve o'clock two minutes and a quarter",
      book: "Crundle Castle",
      author: "Lewis Carroll",
      prefix: "It had struck ",
      suffix:
        '. The Baron\'s footman hastily seized a large goblet, and gasped with terror as he filled it with hot, spiced wine. "Tis past the hour, \'tis past," he groaned in anguish, "and surely I shall now get the red hot poker the Baron hath so often promised me, oh! Woe is me! Would that I had prepared the Baron\'s lunch before!"',
    },
  ],
  "12:03": [
    {
      time: "12.03",
      book: "The Yiddish Policemen's Union",
      author: "Michael Chabon",
      prefix: "At ",
      suffix:
        " the sun has already punched its ticket. Sinking, it stains the cobbles and stucco of the platz in a violin-coloured throb of light that you would have to be a stone not to find poignant.",
    },
  ],
  "12:04": [
    {
      time: "12.04pm",
      book: "The Mezzanine",
      author: "Nicholson Baker",
      prefix: "Though by then it was by Tina's own desk clock ",
      suffix:
        ' I was always touched when, out of a morning\'s worth of repetition, secretaries continued to answer with good mornings for an hour or so into the afternoon, just as people often date things with the previous year well into February; sometimes they caught their mistake and went into a "This is not my day" or "Where is my head?" escape routine; but in a way they were right, since the true tone of afternoons does not take over in offices until nearly two.',
    },
  ],
  "12:05": [
    {
      time: "around noon",
      book: "Odalisque: The Baroque Cycle #3",
      author: "Neal Stephenson",
      prefix: "A few minutes' light ",
      suffix:
        " is all that you need to discover the error, and re-set the clock – provide that you bother to go up and make the observation.",
    },
  ],
  "12:06": [
    {
      time: "around noon",
      book: "Odalisque: The Baroque Cycle #3",
      author: "Neal Stephenson",
      prefix: "A few minutes' light ",
      suffix:
        " is all that you need to discover the error, and re-set the clock – provide that you bother to go up and make the observation.",
    },
  ],
  "12:07": [
    {
      time: "seven minutes after 12",
      book: "The Chronicle of Young Satan",
      author: "Mark Twain",
      prefix: "On a Monday Simon Hirsch was going to break his leg at ",
      suffix:
        ", noon, and as soon as Satan told us the day before, Seppi went to betting with me that it would not happen, and soon they got excited and went to betting with me themselves.",
    },
  ],
  "12:08": [
    {
      time: "12:08",
      book: "Eighty Days",
      author: "Matthew Goodman",
      prefix: "When a clock struck noon in Washington, D.C., the time was ",
      suffix: " in Philadephia, 12:12 in new York, and 12:24 in Boston.",
    },
  ],
  "12:10": [
    {
      time: "noon, and ten minutes later",
      book: "Oracle Night",
      author: "Paul Auster",
      prefix: "Madame Dumas arrived at ",
      suffix:
        " Trause handed her his ATM card and instructed her to go to the neighborhood Citibank near Sheridan Square and transfer forty thousand dollars from his savings account to his checking account.",
    },
    {
      time: "twelve-ten",
      book: "Watchers",
      author: "Dean Koontz",
      prefix:
        "They paid for only one room and kept Einstein with them because they were not going to need privacy for lovemaking. Exhausted, Travis barely managed to kiss Nora before falling into a deep sleep. He dreamed of things with yellow eyes, misshapen heads, and crocodile mouths full of sharks’ teeth. He woke five hours later, at ",
      suffix: " Thursday afternoon.",
    },
  ],
  "12:11": [
    {
      time: "12:11",
      book: "Boy A",
      author: "Jonathan Trigell",
      prefix: "At ",
      suffix:
        ' there was a knock on the door. It was Terry, A could tell. He hadn\'t known Terry long, but there was something calmer, more patient, that separated Terry\'s knocks from the rest of the staff. He knocked from genuine politeness, not formality. "Come in," A said, although the lock was on the other side. Terry did. "It\'s your mother," he said. "There\'s no easy way to say this." Though he had just used the easiest, because A now knew the rest. A’s face froze, as it tried to catch up, as it tried to register the news. Then it crumpled, and while he considered this fresh blow, the tears came.',
    },
  ],
  "12:12": [
    {
      time: "12:12",
      book: "Boy A",
      author: "Jonathan Trigell",
      prefix:
        "It was the twelfth of December, the twelfth month. A was twelve. The electric clock/radio by his bedside table said 12:01. A was waiting for it to read ",
      suffix:
        ", he hoped there would be some sense of cosmic rightness when it did.",
    },
  ],
  "12:14": [
    {
      time: "twelve-fourteen",
      book: "The Plymouth Express",
      author: "Agatha Christie",
      prefix: "She left London on the ",
      suffix:
        " from Paddington, arriving at Bristol (where she had to change) at two-fifty.",
    },
  ],
  "12:15": [
    {
      time: "quarter past twelve",
      book: "The Little Nugget",
      author: "P.G. Wodehouse",
      prefix:
        "Very well, dear,' she said. 'I caught the 10.20 to Eastnor, which isn't a bad train, if you ever want to go down there. I arrived at a ",
      suffix:
        ", and went straight up to the house--you've never seen the house, of course? It's quite charming--and told the butler that I wanted to see Mr Ford on business. I had taken the precaution to find out that he was not there. He is at Droitwich.'",
    },
    {
      time: "12.15",
      book: "A Writer's Diary: Being Extracts from the Diary of Virgina Woolf",
      author: "Virginia Woolf",
      prefix:
        "What shall I think of that's liberating and refreshing? I'm in the mood when I open my window at night and look at the stars. Unfortunately it's ",
      suffix: " on a grey dull day, the aeroplanes are active",
    },
  ],
  "12:17": [
    {
      time: "seventeen minutes after twelve",
      book: "Vanvild Kava",
      author: "Isaac Bashevis Singer",
      prefix:
        "Kava ordered two glasses of coffee for himself and his beloved and some cake. When the pair left, exactly ",
      suffix: ", the club began to buzz with excitement.",
    },
  ],
  "12:20": [
    {
      time: "twelve-twenty",
      book: "Watchers",
      author: "Dean Koontz",
      prefix: "By ",
      suffix:
        " in the afternoon, Vince was seated in a rattan chair with comfortable yellow and green cushions at a table by the windows in that same restaurant. He’d spotted Haines on entering. The doctor was at another window table, three away from Vince, half-screened by a potted palm. Haines was eating shrimp and drinking margaritas with a stunning blonde. She was wearing white slacks and a gaily striped tube-top, and half the men in the place were staring at her.",
    },
    {
      time: "12:20",
      book: "The Day Lady Died",
      author: "Frank O'Hara",
      prefix: "It is ",
      suffix:
        " in New York a Friday three days after Bastille day, yes it is 1959 and I go get a shoeshine because I will get off the 4:19 in Easthampton at 7:15 and then go straight to dinner and I don’t know the people who will feed me",
    },
  ],
  "12:21": [
    {
      time: "Twelve twenty-one",
      book: "11/22/63",
      author: "Stephen King",
      prefix: "Jake think of something. PLEASE! ",
      suffix: ".",
    },
  ],
  "12:22": [
    {
      time: "twenty-two minutes past twelve",
      book: "Narrative of a Journey round the Dead Sea and in the Bible lands in 1850 and 1851",
      author: "Félicien de Saulcy",
      prefix: "By ",
      suffix:
        " we leave, much too soon for our desires, this delightful spot, where the pilgrims are in the habit of bathing who come to visit the Jordan.",
    },
  ],
  "12:24": [
    {
      time: "12:24",
      book: "The Undomestic Goddess",
      author: "Sophie Kinsella",
      prefix: "",
      suffix:
        " My legs are in total agony. I've been kneeling on hard tiles, cleaning the bath, for what seems like hours. There are little ridges where the tiles have dug into my knees, and I'm boiling hot and the cleaning chemicals are making me cough. All I want is a rest. But I can't stop for a moment. I am so behind ..",
    },
  ],
  "12:25": [
    {
      time: "12.25",
      book: "Ulysses",
      author: "James Joyce",
      prefix: "Boys, do it now. God's time is ",
      suffix: ".",
    },
  ],
  "12:26": [
    {
      time: "26.",
      book: "A Kestrel For a Knave",
      author: "Barry Hines",
      prefix: "12.25pm. ",
      suffix: " 27. Every time Billy saved a shot he looked heartbroken",
    },
  ],
  "12:27": [
    {
      time: "27.",
      book: "A Kestrel For a Knave",
      author: "Barry Hines",
      prefix: "12.25pm. 26. ",
      suffix: " Every time Billy saved a shot he looked heartbroken",
    },
  ],
  "12:28": [
    {
      time: "12.28",
      book: "11/22/63",
      author: "Stephen King",
      prefix: "The DRINK CHEER-UP COFFEE wall clock read ",
      suffix: ".",
    },
  ],
  "12:30": [
    {
      time: "half-past twelve",
      book: "Love in a Cold Climate",
      author: "Nancy Mitford",
      prefix:
        "\"You'll never believe this but (in Spain) they are two hours late for ever meal - two hours Fanny - (can we lunch at ",
      suffix: ' today?)"',
    },
    {
      time: "12.30 p.m.",
      book: "The Curious Incident of the Dog in the Night-time",
      author: "Mark Haddon",
      prefix: "",
      suffix: " Lunc",
    },
    {
      time: "half past twelve",
      book: "Northanger Abbey",
      author: "Jane Austen",
      prefix: "At ",
      suffix:
        ", when Catherine’s anxious attention to the weather was over and she could no longer claim any merit from its amendment, the sky began voluntarily to clear. A gleam of sunshine took her quite by surprise; she looked round; the clouds were parting, and she instantly returned to the window to watch over and encourage the happy appearance. Ten minutes more made it certain that a bright afternoon would succeed, and justified the opinion of Mrs. Allen, who had “always thought it would clear up.”",
    },
    {
      time: "12.30pm",
      book: "Fear and Loathing in Las Vegas",
      author: "Hunter S. Thompson",
      prefix: "Tuesday, ",
      suffix:
        "… Baker, California… Into the Ballantine Ale now, zombie drunk and nervous. I recognize this feeling: three or four days of booze, drugs, sun, no sleep and burned out adrenalin reserves – a giddy, quavering sort of high that means the crash is coming. But when? How much longer?",
    },
  ],
  "12:32": [
    {
      time: "12:32",
      book: "The Undomestic Goddess",
      author: "Sophie Kinsella",
      prefix:
        "12:30 What is wrong with this bleach bottle? Which way is the nozzle pointing, anyway? I'm turning it round in confusion, peering at the arrows on the plastic ... Why won't anything come out? OK, I'm going to squeeze it really, really hard- That nearly got my eye. ",
      suffix: " FUCK. What has it done to my HAIR?",
    },
    {
      time: "twelve thirty-two",
      book: "Extremely Entertaining Short Stories",
      author: "Stacy Aumonier",
      prefix:
        "A chutney-biting brigadier named Boyd-Boyd fixed an appointment on the 'phone with Oxted, at Hornborough Station, for the ",
      suffix: ". He was to deliver the goods.",
    },
  ],
  "12:33": [
    {
      time: "12.33",
      book: "Five Red Herrings",
      author: "Dorothy L. Sayers",
      prefix: "It's ",
      suffix:
        " now and I could do it, the station is just down that side road there.",
    },
  ],
  "12:35": [
    {
      time: "twelve-thirty-five",
      book: "Men At Arms",
      author: "Evelyn Waugh",
      prefix:
        "As surely as Apthorpe was marked for early promotion, Trimmer was marked for ignominy. That morning he had appeared at the precise time stated in orders. Everyone else had been waiting five minutes and Colour Sergeant Cork called out the marker just as Trimmer appeared. So it was ",
      suffix: " when they were dismissed.",
    },
  ],
  "12:39": [
    {
      time: "thirty-nine minutes past twelve",
      book: "The Toilers of the Sea",
      author: "Victor Hugo",
      prefix:
        "Next, he remembered that the morrow of Christmas would be the twenty-seventh day of the moon, and that consequently high water would be at twenty-one minutes past three, the half-ebb at a quarter past seven, low water at thirty-three minutes past nine, and half flood at ",
      suffix: ".",
    },
  ],
  "12:40": [
    {
      time: "twenty minutes to one",
      book: "Extremely Entertaining Short Stories (The Octave of Jealousy)",
      author: "Stacy Aumonier",
      prefix: "A little ormolu clock in the outer corridor indicated ",
      suffix:
        ". The car was due at one-fifteen. Thirty-five minutes: oh, to escape for only that brief period!",
    },
  ],
  "12:42": [
    {
      time: "eighteen minutes to one",
      book: "Marjorie Morningstar",
      author: "Herman Wouk",
      prefix:
        "The butt had been growing warm in her fingers; now the glowing end stung her skin. She crushed the cigarette out and stood, brushing ash from her black skirt. It was ",
      suffix:
        ". She went to the house phone and called his room. The telephone rang and rang, but there was no answer.",
    },
  ],
  "12:43": [
    {
      time: "Twelve-forty-three",
      book: "A Pocket Full of Rye",
      author: "Agatha Christie",
      prefix:
        "Died five minutes ago, you say? he asked. His eye went to the watch on his wrist. ",
      suffix: ", he wrote on the blotter.",
    },
  ],
  "12:44": [
    {
      time: "around quarter to one",
      book: "Long Day's Journey Into Night",
      author: "Eugene O'Neil",
      prefix: "It is ",
      suffix:
        ". No sunlight comes into the room now through the windows at right. Outside the day is fine but increasingly sultry, with a faint haziness in the air which softens the glare of the sun.",
    },
  ],
  "12:45": [
    {
      time: "12:45",
      book: "Dracula",
      author: "Bram Stoker",
      prefix:
        'The boy handed in a dispatch. The Professor closed the door again, and after looking at the direction, opened it and read aloud. "Look out for D. He has just now, ',
      suffix:
        ', come from Carfax hurriedly and hastened towards the South. He seems to be going the round and may want to see you: Mina"',
    },
  ],
  "12:46": [
    {
      time: "around quarter to one",
      book: "Long Day's Journey Into Night",
      author: "Eugene O'Neil",
      prefix: "It is ",
      suffix:
        ". No sunlight comes into the room now through the windows at right. Outside the day is fine but increasingly sultry, with a faint haziness in the air which softens the glare of the sun.",
    },
  ],
  "12:49": [
    {
      time: "12:49 hours",
      book: "Bomber",
      author: "Len Deighton",
      prefix: "The first victim of the Krefeld raid died at ",
      suffix:
        " Double British Summer Time at B Flight, but it wasn't due to carelessness.",
    },
  ],
  "12:50": [
    {
      time: "ten minutes to one",
      book: "The Ragged Trousered Philanthropists",
      author: "Robert Tressell",
      prefix:
        "So presently Bert was sent up to the top of the house to look at a church clock which was visible therefrom, and when he came down he reported that it was ",
      suffix: ".",
    },
  ],
  "12:52": [
    {
      time: "12.52",
      book: "Dreams of Leaving",
      author: "Rupert Thomson",
      prefix: "The nightclub stood on the junction, flamboyant, still. It was ",
      suffix: ".",
    },
  ],
  "12:53": [
    {
      time: "12:53",
      book: "Five Red Herrings",
      author: "Dorothy L. Sayers",
      prefix:
        "Aboot twelve miles. We ought tae pass her at Pinmore. She's due there at ",
      suffix: ".",
    },
  ],
  "12:54": [
    {
      time: "12:54 pm.",
      book: "Varieties of Disturbance",
      author: "Lydia Davis",
      prefix:
        "I listen to the different boats' horns, hoping to learn what kind of boat I'm hearing and what the signal means: is the boat leaving or entering the harbor; is it the ferry, or a whale-watching boat, or a fishing boat? At 5:33 pm there is a blast of two deep, resonant notes a major third apart. On another day there is the same blast at ",
      suffix: " On another, exactly 8:00 am.",
    },
  ],
  "12:55": [
    {
      time: "Five to one",
      book: "A Man Lay Dead",
      author: "Ngaio Marsh",
      prefix: "The inspector glanced at the clock. ",
      suffix: ". A busy morning.",
    },
  ],
  "12:58": [
    {
      time: "12.58pm",
      book: "Magic Bites",
      author: "Ilona Andrews",
      prefix: "The watch on my wrist showed ",
      suffix: " I'd have time to hit the morgue.",
    },
  ],
  "12:59": [
    {
      time: "12.59pm",
      book: "The Curious Incident of the Dog in the Night-Time",
      author: "Mark Haddon",
      prefix:
        "And I had been looking at my watch since the train had started at ",
      suffix: "",
    },
  ],
  "13:00": [
    {
      time: "clock strikes one",
      book: "The Scarlet Pimpernel",
      author: "Baroness Orczy",
      prefix:
        '"I think," he said, with a triumphant smile, "that I may safely expect to find the person I seek in the dining-room, fair lady." "There may be more than one." "Whoever is there, as the ',
      suffix:
        ", will be shadowed by one of my men; of these, one, or perhaps two, or even three, will leave for France to-morrow. One of these will be the `Scarlet Pimpernel.'\"",
    },
    {
      time: "One o'clock",
      book: "Jingo",
      author: "Terry Pratchett",
      prefix: '"',
      suffix: ' pee em! Hello, Insert Name Here!" Said by the Disorganizer',
    },
    {
      time: "one o'clock",
      book: "Dracula",
      author: "Bram Stoker",
      prefix: "“Czarina Catherine reported entering Galatz at ",
      suffix: " today.”",
    },
    {
      time: "1.00 p.m.",
      book: "The Curious Incident of the Dog in the Night-time",
      author: "Mark Haddon",
      prefix: "",
      suffix: " First afternoon clas",
    },
    {
      time: "1 o'clock",
      book: "Girl, Interrupted",
      author: "Susanna Kaysen",
      prefix: "After ",
      suffix: " checks, Gretta always goes out for a smoke.",
    },
    {
      time: "1pm",
      book: "Platform",
      author: "Michel Houellebecq",
      prefix: "Gottfried Rembke arrived at ",
      suffix:
        " precisely. The moment he walked into the restaurant, handed his coat to the waiter, they knew it was him. The solid, stocky body, the gleaming pate, the open expression, the vigorous handshake: everything about him radiated ease and enthusiasm",
    },
    {
      time: "one o'clock",
      book: "Jigsaw",
      author: "Sybille Bedford",
      prefix: "I got to Schmidt's early, feeling horribly nervous. At ",
      suffix:
        " sharp: Toni. She was looking at the menu she knew well - Schmorbraten? Schnitzel? - when he loomed over her. I had seen him come in. She looked up, through him, at me. 'Traitor.' Jamie, hovering, looking very big, said her pet name, a German diminutive chosen by her. Toni addressed the air. 'If he does not leave at once I shall tell the waiter that I am not sharing my table with this gentleman.' Jamie heard, said her name again, turned to go, I rose to go with him. Toni - with that concentration of will - said, 'YOU are lunching with me.'",
    },
    {
      time: "thirteen",
      book: "Nineteen Eighty-Four",
      author: "George Orwell",
      prefix:
        "It was a bright cold day in April, and the clocks were striking ",
      suffix: ".",
    },
    {
      time: "one o'clock",
      book: "The Swimmer",
      author: "Roma Tearne",
      prefix: "It was ",
      suffix:
        ". I bought some apples and a small pork pie and drove across the bridge to the other side of the riverbank in the direction of Orford Ness.",
    },
    {
      time: "at one",
      book: "Silly Old Baboon",
      author: "Spike Milligan",
      prefix:
        "Many moons passed by. Did Baboon ever fly? Did he ever get to the sun? I’ve just heard today That he’s well on his way! He’ll be passing through Acton ",
      suffix: ".",
    },
    {
      time: "one o'clock",
      book: "Swallows and Amazons",
      author: "Arthur Ransome",
      prefix: "That day it was ",
      suffix:
        " before John and Roger rowed across and went up to Dixon's farm for the milk and a new supply of eggs and butter.",
    },
    {
      time: "one o'clock",
      book: "One Flew Over the Cuckoo's Nest",
      author: "Ken Kesey",
      prefix: "The day-room floor gets cleared of tables and at ",
      suffix:
        " the doctor comes out of his office down the hall, nods once at the nurse as he goes past where he's watching out of her window, sits in his chair just to the left of the door.",
    },
  ],
  "13:01": [
    {
      time: "about one",
      book: "Ulysses",
      author: "James Joyce",
      prefix:
        "There's five fathoms out there, he said. It'll be swept up that way when the tide comes in ",
      suffix: ". It's nine days today.",
    },
  ],
  "13:02": [
    {
      time: "about one o'clock",
      book: "A Clergyman's Daughter",
      author: "George Orwell",
      prefix: "At ",
      suffix: " the overseer arrived and told them he had no jobs for them",
    },
  ],
  "13:03": [
    {
      time: "a little after one o'clock",
      book: "The Big Clock",
      author: "Kenneth Fearing",
      prefix: "It was ",
      suffix:
        " when I got there, time for lunch, so I had it. The food was awful. But it would go on the expense account, and after I'd eaten I got out my notebook and put it down. Lunch $1.50. Taxi $1.00.",
    },
  ],
  "13:04": [
    {
      time: "four minutes past one",
      book: "The Ragged Trousered Philanthropists",
      author: "Robert Tressell",
      prefix: '"Jesus Christ!" he gasped. "It\'s ',
      suffix:
        '!" Linden frantically seized hold of a pair of steps and began wandering about the room with them.',
    },
  ],
  "13:05": [
    {
      time: "five past one",
      book: "The Undomestic Goddess",
      author: "Sophie Kinsella",
      prefix:
        "“Samantha?” I can hear Trish approaching. “Um ... hold on!” I hurry to the door, trying to block her view. “It's already ",
      suffix:
        ",” I can hear her saying a little sharply. “And I did ask, most clearly for ...” Her voice trails off into silence as she reaches the kitchen door, and her whole face sags in astonishment. I turn and follow her gaze as she surveys the endless plates of sandwiches. “My goodness!” At last Trish finds her voice. “This is ... this is very impressive!”",
    },
    {
      time: "five past one",
      book: "A Man Lay Dead",
      author: "Ngaio Marsh",
      prefix: "At ",
      suffix:
        " Alleyn opened the outer door, knocked his pipe out on the edge of the stone step,and remained staring out on to the drive.",
    },
  ],
  "13:06": [
    {
      time: "13 hours and 6 minutes",
      book: "We",
      author: "Yevgeny Zamyatin",
      prefix: "And then at precisely ",
      suffix: " - confusion broke out in the rectangle.",
    },
  ],
  "13:09": [
    {
      time: "nine minutes past one",
      book: "Tortured Souls: The Legend of Primordium",
      author: "Clive Barker",
      prefix: "At ",
      suffix:
        ", a pair of horses approached (not from the city, from which direction Krieger had expected her to come, but from the Desert, which lay, vast and largely uncharted, out to the West and South-West of the city.)",
    },
  ],
  "13:10": [
    {
      time: "ten minutes past one",
      book: "Death on the Nile",
      author: "Agatha Christie",
      prefix: '"It was ',
      suffix: '.” “You are sure of that?"',
    },
  ],
  "13:11": [
    {
      time: "1.11",
      book: "Five Red Herrings",
      author: "Dorothy L. Sayers",
      prefix:
        "I pursued my inquiries at the other stations along the line an' I found there was a gentleman wi' a bicycle tuk the ",
      suffix: " train at Girvan.",
    },
  ],
  "13:13": [
    {
      time: "thirteen minutes past one",
      book: "Journey to the Centre of the Earth",
      author: "Jules Verne",
      prefix:
        '"There it is! There it is!" shouted the Professor. "Now for the centre of the globe!" he added in Danish. I looked at Hans. "Forüt!" was his tranquil answer. "Forward!" replied my uncle. It was ',
      suffix: ".",
    },
  ],
  "13:15": [
    {
      time: "One hour and a quarter",
      book: "The Picture of Dorian Gray",
      author: "Oscar Wilde",
      prefix:
        "‘Monsieur has well slept this morning,’ he said, smiling. ‘What o’clock is it, Victor?’ asked Dorian Gray, sleepily. ‘",
      suffix: ", monsieur.’",
    },
    {
      time: "One-fifteen",
      book: "A Man Lay Dead",
      author: "Ngaio Marsh",
      prefix:
        '"Where are the ladies and Gentlemen?" asked Aleyn. "Sir, in the garding", said Bunce. "What time\'s lunch?" "',
      suffix: '".',
    },
    {
      time: "Quarter-past one",
      book: "Miss Pettigrew lives for a Day",
      author: "Winifred Watson",
      prefix:
        "The clock caught Miss LaFosse´s eye. ´Good heavens!´ she gasped. ´Look at the time. ",
      suffix:
        ". You must be starved.' She turned impetuously to Miss Pettigrew.",
    },
  ],
  "13:16": [
    {
      time: "1.16pm",
      book: "The Curious Incident of the Dog in the Night-Time",
      author: "Mark Haddon",
      prefix: "And the first stop had been at ",
      suffix: " which was 17 minutes later.",
    },
  ],
  "13:17": [
    {
      time: "One seventeen",
      book: "The Terrorist, He Watches",
      author: "Wislawa Szymborska",
      prefix: "",
      suffix:
        " and four seconds. That shorter guy’s really got it made, and gets on a scooter, and that taller one, he goes in. One seventeen and forty seconds. That girl there, she’s got a green ribbon in her hair. Too bad that bus just cut her from view",
    },
  ],
  "13:18": [
    {
      time: "One eighteen",
      book: "The Terrorist, He Watches",
      author: "Wislawa Szymborska",
      prefix: "",
      suffix:
        " exactly. Was she stupid enough to head inside? Or wasn't she? We'll know before long, When the dead are carried out",
    },
  ],
  "13:20": [
    {
      time: "1320 hours",
      book: "The Hunt for Red October",
      author: "Tom Clancy",
      prefix:
        "Kamarov, signal to Purga: 'Diving at—,'\" he checked his watch, \"'—",
      suffix:
        '. Exercise OCTOBER FROST begins as scheduled. You are released to other assigned duties. We will return as scheduled." Kamarov worked the trigger on the blinker light to transmit the message. The Purga responded at once, and Ramius read the flashing signal unaided: "IF THE WHALES DON\'T EAT YOU. GOOD LUCK TO RED OCTOBER!"',
    },
    {
      time: "twenty minutes past one",
      book: "Dracula",
      author: "Bram Stoker",
      prefix:
        "The time is coming for action. Today this Vampire is limit to the powers of man, and till sunset he may not change. It will take him time to arrive here, see it is ",
      suffix:
        ", and there are yet some times before he can hither come, be he never so quick.",
    },
    {
      time: "twenty minutes past one",
      book: "Dracula",
      author: "Bram Stoker",
      prefix:
        "Today this Vampire is limit to the powers of man, and till sunset he may not change. It will take him time to arrive here, see it is ",
      suffix:
        ", and there are yet some times before he can hither come, be he never so quick.",
    },
  ],
  "13:23": [
    {
      time: "1.23pm",
      book: "The Curious Incident Of The Dog In The Night-Time",
      author: "Mark Haddon",
      prefix:
        'And when we got to Swindon Mother had keys to the house and we went in and she said, "Hello?" but there was no one there because it was ',
      suffix: ".",
    },
    {
      time: "twenty-three minutes past one",
      book: "A Mummer's Tale",
      author: "Anatole France",
      prefix: "The clock marked ",
      suffix:
        ". He was suddenly full of agitation, yet hopeful. She had come! Who could tell what she would say? She might offer the most natural explanation of her late arrival. Félicie entered the room, her hair in disorder, her eyes shining, her cheeks white, her bruised lips a vivid red; she was tired, indifferent, mute, happy and lovely, seeming to guard beneath her cloak, which she held wrapped about her with both hands, some remnant of warmth and voluptuous pleasure.",
    },
  ],
  "13:24": [
    {
      time: "1:24 p.m",
      book: "The Cold Six Thousand",
      author: "James Ellroy",
      prefix: "Littell checked his watch - ",
      suffix: " - Littell grabbed the phone by the bed.",
    },
  ],
  "13:25": [
    {
      time: "One-twenty-five",
      book: "Hard Boiled Wonderland and the End of the World",
      author: "Haruki Murakami",
      prefix:
        "I'd really have liked to, I told her, if it weren't for the things I had in the drier. I cast an eye at my watch. ",
      suffix: ". The drier had already stopped.",
    },
  ],
  "13:26": [
    {
      time: "around one-thirty",
      book: "The Stranger",
      author: "Albert Camus",
      prefix: "Raymond came back with Masson ",
      suffix:
        ". His arm was bandaged up and he had an adhesive plaster on the corner of his mouth. The doctor had told him it was nothing, but Raymond looked pretty grim. Masson tried to make him laugh. But he still wouldn't say anything.",
    },
  ],
  "13:30": [
    {
      time: "half-past one",
      book: "The Diary of a Nobody",
      author: "George and Weedon Grossmith",
      prefix: "Lupin not having come down, I went up again at ",
      suffix: ', and said we dined at two; he said he "would be there."',
    },
    {
      time: "half past one",
      book: "Brighton Rock",
      author: "Graham Greene",
      prefix: "She was a sticker. A clock away in the town struck ",
      suffix: ".",
    },
    {
      time: "half-past one",
      book: "Mrs dalloway",
      author: "Virginia Woolf",
      prefix:
        "Shredding and slicing, dividing and subdividing, the clocks of Harley Street nibbled at the June day, counselled submission, upheld authority, and pointed out in chorus the supreme advantages of a sense of proportion, until the mound of time was so far diminished that a commercial clock, suspended above a shop in Oxford Street, announced, genially and fraternally, as if it were a pleasure to Messrs Rigby and Lowndes to give the information gratis, that it was ",
      suffix: ".",
    },
  ],
  "13:32": [
    {
      time: "one ... thirty-two",
      book: "So Long, and Thanks for All the Fish",
      author: "Douglas Adams",
      prefix: "At the third stroke it will be ",
      suffix:
        " ... and twenty seconds. 'Beep ... beep ... beep.' Ford Prefect suppressed a little giggle of evil satisfaction, realized that he had no reason to suppress it, and laughed out loud, a wicked laugh.",
    },
  ],
  "13:33": [
    {
      time: "one ... thirty-three",
      book: "So Long, and Thanks for All the Fish",
      author: "Douglas Adams",
      prefix:
        "He waited for the green light to show and then opened the door again on to the now empty cargo hold.'... ",
      suffix: " ... and fifty seconds.' Very nice.",
    },
  ],
  "13:34": [
    {
      time: "one ... thirty-four",
      book: "So Long, and Thanks for All the Fish",
      author: "Douglas Adams",
      prefix:
        "'At the third stroke it will be ...' He tiptoed out and returned to the control cabin. '... ",
      suffix:
        " and twenty seconds.' The voice sounded as clear as if he was hearing it over a phone in London, which he wasn't, not by a long way.",
    },
    {
      time: "one ... thirty ... four",
      book: "So Long, and Thanks for All the Fish",
      author: "Douglas Adams",
      prefix:
        "He then went and had a last thorough examination of the emergency suspended animation chamber, which was where he particularly wanted it to be heard. 'At the third stroke it will be ",
      suffix: " ... precisely.'",
    },
  ],
  "13:37": [
    {
      time: "1.37pm",
      book: "Light House",
      author: "William Monahan",
      prefix:
        "He had not dared to sleep in his rented car—you didn't sleep in your car when you worked for Jesus Castro—and he was beginning to hallucinate. Still, he was on the job, and he scribbled in his notebook:\" ",
      suffix: ' Subject appears to be getting laid."',
    },
  ],
  "13:39": [
    {
      time: "1.39pm",
      book: "The Curious Incident of the Dog in the Night-Time",
      author: "Mark Haddon",
      prefix: "And it was now ",
      suffix:
        " which was 23 minutes after the stop, which mean that we would be at the sea if the train didn't go in a big curve. But I didn't know if it went in a big curve.",
    },
  ],
  "13:42": [
    {
      time: "1.42pm",
      book: "The Girl with the Dragon Tattoo",
      author: "Stieg Larsson",
      prefix: "The last note was recorded at ",
      suffix: ": G.M. on site at H-by; will take over the matter.",
    },
  ],
  "13:44": [
    {
      time: "forty-four minutes past one",
      book: "Mr. Policeman and the Cook",
      author: "Wilkie Collins",
      prefix: "By good luck, the next train was due at ",
      suffix:
        ", and arrived at Yateland (the next station) ten minutes afterward.",
    },
  ],
  "13:45": [
    {
      time: "quarter to two",
      book: "Mike",
      author: "PG Wodehouse",
      prefix:
        "That period which is always so dangerous, when the wicket is bad, the ten minutes before lunch, proved fatal to two more of the enemy. The last man had just gone to the wickets, with the score at a hundred and thirty-one, when a ",
      suffix: " arrived, and with it the luncheon interval.",
    },
    {
      time: "one forty-five",
      book: "Sir Roderick Comes to Lunch",
      author: "P.G. Wodehouse",
      prefix: "The blow fell at precisely ",
      suffix:
        " (summer-time). Benson, my Aunt Agatha's butler, was offering me the fried potatoes at the moment, and such was my emotion that I lofted six of them on the sideboard with the spoon.",
    },
  ],
  "13:47": [
    {
      time: "1.47pm.",
      book: "The Woman Who Went To Bed For A Year",
      author: "Sue Townsend",
      prefix:
        "Poppy was sprawled on Brianne's bed, applying black mascara to her stubby lashes. Brianne was sitting at her desk, trying to complete an essay before the 2pm deadline. It was ",
      suffix: "",
    },
  ],
  "13:48": [
    {
      time: "twelve minutes to two",
      book: "The Apocalypse Watch",
      author: "Robert Ludlum",
      prefix: "It was ",
      suffix:
        " in the afternoon when Claude Moreau and his most-trusted field officer, Jacques Bergeron, arrived at the Georges Cinq station of the Paris Metro. They walked, separately, to the rear of the platform, each carrying a handheld radio, the frequencies calibrated to each other.",
    },
  ],
  "13:49": [
    {
      time: "1.49",
      book: "Five Red Herrings",
      author: "Dorothy L. Sayers",
      prefix:
        "The bookstall clerk had seen the passenger in grey pass the bookstall at ",
      suffix: ", in the direction of the exit.",
    },
  ],
  "13:50": [
    {
      time: "Ten to two",
      book: "The God of Small Things",
      author: "Arundhati Roy",
      prefix: "Rahel's toy wristwatch had the time painted on it. ",
      suffix:
        ". One of her ambitions was to own a watch on which she could change the time whenever she wanted to (which according to her was what Time was meant for in the first place).",
    },
    {
      time: "one-fifty",
      book: "The Cornish Mystery",
      author: "Agatha Christie",
      prefix: "The best train of the day was the ",
      suffix:
        " from Paddington which reached Polgarwith just after seven o'clock.",
    },
  ],
  "13:55": [
    {
      time: "five minutes before two",
      book: "The Professor",
      author: "Charlotte Brontë",
      prefix:
        "If I was punctual in quitting Mlle. Reuter's domicile, I was at least equally punctual in arriving there; I came the next day at ",
      suffix:
        ', and on reaching the schoolroom door, before I opened it, I heard a rapid, gabbling sound, which warned me that the "priere du midi" was not yet concluded.',
    },
  ],
  "13:57": [
    {
      time: "three minutes to two",
      book: "Urban Shaman",
      author: "C.E. Murphy",
      prefix: "I looked for a clock. It was ",
      suffix:
        '. "I hope you can catch him, then. Thank you. I really appreciate it."',
    },
  ],
  "13:58": [
    {
      time: "almost two o’clock",
      book: "King of Tuzla",
      author: "Arnold Jansen op de Haar",
      prefix: "It was ",
      suffix:
        ", but nothing moved, Stari Teočak was silent and so empty it seemed abandoned, and yet Tijmen constantly felt he was being observed by invisible eyes.",
    },
  ],
  "13:59": [
    {
      time: "One ... fifty-nine …",
      book: "So Long, and Thanks for All the Fish",
      author: "Douglas Adams",
      prefix:
        "For twenty minutes he sat and watched as the gap between the ship and Epun closed, as the ship's computer teased and kneaded the numbers that would bring it into a loop around the little moon, and close the loop and keep it there, orbiting in perpetual obscurity. '",
      suffix: "'\"",
    },
  ],
  "14:00": [
    {
      time: "two o'clock",
      book: "A Confederacy of Dunces",
      author: "John Kennedy Toole",
      prefix:
        "'She could have fired the jig, and he could have kept on picking up his packages at the old time, ",
      suffix: ". As it was, he had almost been arrested.'",
    },
    {
      time: "two o'clock",
      book: "The Outsider",
      author: "Albert Camus",
      prefix:
        "\"The old people's home is at Marengo, fifty miles from Algiers. I'll catch the ",
      suffix:
        ' bus and get there in the afternoon.".... "I caught the two o\'clock bus. It was very hot."',
    },
    {
      time: "1400 hours",
      book: "Black Swan Green",
      author: "David Mitchell",
      prefix: "At approximately ",
      suffix:
        " a pair of enemy Skyhawks came flying in at deck level out of nowhere.",
    },
    {
      time: "two o'clock",
      book: "The Great Gatsby",
      author: "F. Scott Fitzgerald",
      prefix: "At ",
      suffix:
        " Gatsby put on his bathing suit and left word with the butler that if any one phoned word was to be brought to him at the pool.",
    },
    {
      time: "At two",
      book: "The Snowman",
      author: "Jo Nesbo",
      prefix: "",
      suffix: ", the snowplows were in action in Lillestrom",
    },
    {
      time: "two o'clock",
      book: "The Outsider",
      author: "Albert Camus",
      prefix: "I caught the ",
      suffix:
        " bus. It was very hot. I ate at Céleste's restaurant as usual. They all felt very sorry for me and Céleste told me, 'There's no one like a mother'.",
    },
    {
      time: "two o'clock",
      book: "The Outsider",
      author: "Albert Camus",
      prefix:
        "The Home for Aged Persons is at Marengo, some fifty miles from Algiers. With the ",
      suffix:
        " bus I should get there well before nightfall. Then I can spend the night there, keeping the usual vigil beside the body, and be back here by tomorrow evening.",
    },
    {
      time: "2.00",
      book: "The Girl Who Kicked the Hornets' Nest",
      author: "Stieg Larsson",
      prefix: "When Salander woke up it was ",
      suffix: " on Saturday afternoon and a doctor was poking at her.",
    },
  ],
  "14:01": [
    {
      time: "about two o' clock",
      book: "A Single Pebble",
      author: "John Hershey",
      prefix: "At ",
      suffix:
        " the owners young wife came, carrying a handleless cup and a pot with a quilted cover, to where I was still lying disconsolate",
    },
    {
      time: "about two o'clock",
      book: "A Month in the Country",
      author: "JL Carr",
      prefix:
        "The next day was Saturday and, now that Moon was done, I decided to bring the job to its end. So I sent word that I shouldn't be able to umpire for the team at Steeple Sinderby and, after working through the morning, came down ",
      suffix: ".",
    },
  ],
  "14:02": [
    {
      time: "14.02",
      book: "The Woman Who Died A Lot",
      author: "Jasper Fforde",
      prefix: '"I\'m not dead. How did that happen?" He was right. It was ',
      suffix:
        " and twenty-six seconds. Destiny had not been fulfilled. We all looked at each other, confused.",
    },
  ],
  "14:04": [
    {
      time: "2.04pm.",
      book: "The Night of the Generals",
      author: "Hans Hellmut Kirst",
      prefix: "",
      suffix:
        " Once again, the Quartermaster-General's office came on the line asking for Colonel Finckh, and once again Finckh heard the quiet, unemotional, unfamiliar voic",
    },
  ],
  "14:05": [
    {
      time: "five past two",
      book: "A Country Doctor's Notebook",
      author: "Mikhail Bulgakov",
      prefix: "...and at ",
      suffix:
        " on 17 September of that same unforgettable year 1916, I was in the Muryovo hospital yard, standing on trampled withered grass, flattened by the September rain.",
    },
  ],
  "14:06": [
    {
      time: "six minutes past two",
      book: "A Change of Climate",
      author: "Hilary Mantel",
      prefix:
        "A man driving a tractor saw her, four hundred yards from her house, ",
      suffix: " in the afternoon.",
    },
  ],
  "14:10": [
    {
      time: "ten past two",
      book: "The Coward's Tale",
      author: "Vanessa Gebbie",
      prefix:
        'Mrs Eunice Harris pulls back the sleeve of her good coat and checks her good watch. "Indeed yes. Half twelve," and waves a hand at the Town Hall clock as if it was hers. "Always ',
      suffix: '. Someone put a nail in the time years back."',
    },
  ],
  "14:13": [
    {
      time: "two ... thirteen",
      book: "So Long, and Thanks for All the Fish",
      author: "Douglas Adams",
      prefix: "At the third stroke, it will be ",
      suffix: " ... and fifty seconds.'",
    },
  ],
  "14:15": [
    {
      time: "2.15 P.M.",
      book: "Lolita",
      author: "Vladimir Nabokov",
      prefix: "I had a date with her next day at ",
      suffix:
        " In my own rooms, but it was less successful, she seemed to have grown less juvenile, more of a woman overnight.",
    },
    {
      time: "2.15 p.m.",
      book: "The Curious Incident of the Dog in the Night-time",
      author: "Mark Haddon",
      prefix: "",
      suffix: " Second afternoon clas",
    },
    {
      time: "2.15pm",
      book: "Lolita",
      author: "Vladimir Nabokov",
      prefix: "I had a date with her next day at ",
      suffix:
        " in my own rooms, but it was less successful, she seemed to have grown less juvenile, more of a woman overnight. A cold I caught from her led me to cancel a fourth assignment, nor was I sorry to break an emotional series that threatened to burden me with heart-rending fantasies and peter out in dull disappointment. So let her remain, sleek, slender Monique, as she was for a minute or two",
    },
  ],
  "14:16": [
    {
      time: "2.16",
      book: "Five Red Herrings",
      author: "Dorothy L. Sayers",
      prefix:
        "Oh, good evening. I think you were on the barrier when I came in at ",
      suffix:
        " this afternoon. Now, do you know that you let me get past without giving up my ticket? Yes, yes he-he! I really think you ought to be more careful",
    },
  ],
  "14:19": [
    {
      time: "2:19",
      book: "The Cold Six Thousand",
      author: "James Ellroy",
      prefix: "",
      suffix:
        ": Duane Hinton walks out. He walks through the backyard. He lugs some clothes. He wore said clothes last night. He walks to the fence. He feeds the incinerator. He lights a match",
    },
  ],
  "14:20": [
    {
      time: "twenty minutes past two",
      book: "A tale of two cities",
      author: "Charles Dickens",
      prefix:
        "The having originated a precaution which was already in course of execution, was a great relief to Miss Pross. The necessity of composing her appearance so that it should attract no special notice in the streets, was another relief. She looked at her watch, and it was ",
      suffix: ". She had no time to lose, but must get ready at once.",
    },
    {
      time: "Twenty past two",
      book: "The Doctor's Wife",
      author: "Brian Moore",
      prefix:
        "Inevitable, implacable, the rainstorm wept itself out. She saw Tom look at his watch. 'What time is it?' '",
      suffix:
        ". Want to go back to the hotel for a while?' 'All right.' They walked out of the gardens and down the rue de Vaugirard. This holiday, unlike those holidays long ago, would not end with her sleeping at home. Two nights from now I will be high over the Atlantic Ocean and on Saturday I will be walking around in the Other Place. I am going to America. I am starting my life over again. But as she said these words to herself, she found it hard to imagine what the new life would be like. And, again, she was afraid.",
    },
    {
      time: "twenty minutes past two",
      book: "A Tale of Two Cities",
      author: "Charles Dickens",
      prefix: "She looked at her watch and it was ",
      suffix: ". She had no time to lose but must get ready at once.",
    },
    {
      time: "twenty minutes past two",
      book: "The Mystery of Edwin Drood",
      author: "Charles Dickens",
      prefix:
        "The watch found at the Weir was challenged by the jeweller as one he had wound and set for Edwin Drood, at ",
      suffix:
        " on that same afternoon; and it had run down, before being cast into the water; and it was the jeweller's positive opinion that it had never been re-wound.",
    },
  ],
  "14:22": [
    {
      time: "Two-twenty-two",
      book: "Larry's Party",
      author: "Carol Shields",
      prefix:
        "Garth here. Sunday afternoon. Sorry to miss you, but I'll leave a brief message on your tape. ",
      suffix: " or there-aboutish. Great party.",
    },
  ],
  "14:25": [
    {
      time: "2:25",
      book: "The Corrections",
      author: "Jonathan Franzen",
      prefix:
        "Gary shut himself inside his office and flipped through the messages. Caroline had called at 1:35, 1:40, 1:50, 1:55, and 2:10; it was now ",
      suffix:
        ". He pumped his fist in triumph. Finally, finally, some evidence of desperation.",
    },
  ],
  "14:28": [
    {
      time: "28 minutes and 57 seconds after 2pm",
      book: "Ratner's Star",
      author: "Don DeLillo",
      prefix:
        "It happened to be the case that the sixty-based system coincided with our our current method of keeping time... Apparently they wanted us to know that that something might happen at ",
      suffix: " on a day yet to be specified.",
    },
  ],
  "14:30": [
    {
      time: "2:30",
      book: "Maus",
      author: "Art Spiegelman",
      prefix: "Ach! It's ",
      suffix:
        ". Look how the time is flying. And it's still so much to do today.. It's dishes to clean, dinner to defrost, and my pills I haven't yet counted. I don't get it... Why didn't the Jews at least try to resist? It wasn't so easy like you think. Everybody was so starving and frightened, and tired they couldn't believe even what's in front of their eyes.",
    },
    {
      time: "2.30pm",
      book: "The Wind on the Moon",
      author: "Eric Linklater",
      prefix: "At ",
      suffix:
        " on the 13th inst. began to shadow Sir Bobadil the Ostrich, whom I suspect of being the criminal. Shadowing successful. Didn't lose sight of him once.",
    },
    {
      time: "half past two",
      book: "Corker's Freedom",
      author: "John Berger",
      prefix: "At ",
      suffix:
        " the same afternoon the boy and the elderly man are standing in the room directly above the Inner Office and Waiting-Room.",
    },
    {
      time: "half-past two",
      book: "Millie",
      author: "Katherine Mansfield",
      prefix: "It was ",
      suffix:
        " in the afternoon. The sun hung in the faded blue sky like a burning mirror, and away beyond the paddocks the blue mountains quivered and leapt like sea. Sid wouldn't be back until half-past ten. He had ridden over to the township with four of the boys to help hunt down the young fellow who'd murdered Mr. Williamson. Such a dreadful thing!",
    },
    {
      time: "half-past two",
      book: "Dracula",
      author: "Bram Stoker",
      prefix: "It was ",
      suffix:
        ' o\'clock when the knock came. I took my courage a deux mains and waited. In a few minutes Mary opened the door, and announced "Dr. Van Helsing".',
    },
    {
      time: "1/2 past 2 o'clock",
      book: "The Journals of Dorothy Wordsworth",
      author: "Dorothy Wordsworth",
      prefix:
        "May 14th 1800. Wm and John set off into Yorkshire after dinner at ",
      suffix:
        ", cold pork in their pockets. I left them at the turning of the Low-wood bay under the trees. My heart was so full that I could barely speak to W. when I gave him a farewell kiss.",
    },
  ],
  "14:32": [
    {
      time: "2.32 p.m.",
      book: "Post Office",
      author: "Charles Bukowski",
      prefix: "Like ",
      suffix:
        ", Beecher and Avalon, L3 R2 (which meant left three blocks, right two) 2:35 p.m., and you wondered how you could pick up one box, then drive 5 blocks in 3 minutes and be finished cleaning out another box.",
    },
  ],
  "14:36": [
    {
      time: "Two thirty-six",
      book: "The Elephant Vanishes",
      author: "Haruki Murakami",
      prefix: "I look at my watch. ",
      suffix:
        ". All I've got left today is take in the laundry and fix dinner.",
    },
  ],
  "14:39": [
    {
      time: "2.39",
      book: "Five Red Herrings",
      author: "Dorothy L. Sayers",
      prefix:
        "Noo, there's a report come in fra' the station-master at Pinwherry that there was a gentleman tuk the ",
      suffix: " at Pinwherry.",
    },
  ],
  "14:40": [
    {
      time: "two-forty",
      book: "The Catcher in the Rye",
      author: "J.D. Salinger",
      prefix:
        "If a girl looks swell when she meets you, who gives a damn when she's late? 'We better hurry', I said. 'The show starts at ",
      suffix: ".'",
    },
    {
      time: "twenty minutes to three",
      book: "Sinister Street",
      author: "Compton Mackenzie",
      prefix:
        "Members of Big Side marked Michael and Alan as the two most promising three-quarters for Middle Side next year, and when the bell sounded at ",
      suffix:
        ", the members of Big Side would walk with Michael and Alan towards the changing room and encourage them by flattery and genial ragging.",
    },
  ],
  "14:41": [
    {
      time: "2.41",
      book: "Miss Pym Disposes",
      author: "Josephine Tey",
      prefix: "At ",
      suffix:
        ", when the afternoon fast train to London was pulling out of Larborough prompt to the minute, Miss Pym sat under the cedar on the lawn wondering whether she was a fool, and not caring much anyhow.",
    },
  ],
  "14:43": [
    {
      time: "2.43pm",
      book: "Now: Zero",
      author: "JG Ballard",
      prefix: "Jacobson died at ",
      suffix:
        " the next day after slashing his wrists with a razor blade in the second cubicle from the left in the men's washroom on the third floor.",
    },
  ],
  "14:45": [
    {
      time: "quarter to three",
      book: "The Diary of a Nobody",
      author: "George and Weedon Grossmith",
      prefix: "He never came down till a ",
      suffix: ".",
    },
    {
      time: "two forty-five",
      book: "The Pregnant Window",
      author: "Martin Amis",
      prefix:
        "Pull the other one, and tell it to the marines, and don't make me laugh, and fuck off out of it, and all that, but the fact remained that it was still only ",
      suffix: "'.",
    },
    {
      time: "quarter to three",
      book: "The Old Woman",
      author: "Daniil Ivanovich Kharms",
      prefix:
        "What time is it?' 'Look for yourself,' the old woman says to me. I look, and I see the clock has no hands. 'There are no hands,' I say. The old woman looks at the clock face and says to me, 'It's a ",
      suffix: "'.",
    },
  ],
  "14:50": [
    {
      time: "ten to three",
      book: "The Old Vicarage, Grantchester",
      author: "Rupert Brooke",
      prefix: "Stands the Church clock at ",
      suffix: "? And is there honey still for tea?",
    },
  ],
  "14:54": [
    {
      time: "about 2.55",
      book: "Life, the universe and everything",
      author: "Douglas Adams",
      prefix:
        "In the end, it was the Sunday afternoons he couldn’t cope with, and that terrible listlessness that starts to set in ",
      suffix:
        ", when you know you’ve had all the baths you can usefully have that day, that however hard you stare at any given paragraph in the newspaper you will never actually read it, or use the revolutionary new pruning technique it describes, and that as you stare at the clock the hands will move relentlessly on to four o’clock, and you will enter the long dark teatime of the soul.",
    },
  ],
  "14:55": [
    {
      time: "Five to three",
      book: "Ulysses",
      author: "James Joyce",
      prefix:
        "The superior, the very reverend John Conmee SJ reset his smooth watch in his interior pocket as he came down the presbytery steps. ",
      suffix: ". Just nice time to walk to Artane.",
    },
  ],
  "14:56": [
    {
      time: "2.56 P.M.",
      book: "The 60 Minute Zoom",
      author: "JG Ballard",
      prefix: "",
      suffix:
        " Helen is alone now. Her face is out of frame, and through the viewfinder I see only a segment of the pillow, an area of crumpled sheet and the upper section of her chest and shoulders",
    },
  ],
  "14:58": [
    {
      time: "two minutes to three",
      book: "Sinister Street",
      author: "Compton Mackenzie",
      prefix:
        "From twenty minutes past nine until twenty-seven minutes past nine, from twenty-five minutes past eleven until twenty-eight minutes past eleven, from ten minutes to three until ",
      suffix:
        " the heroes of the school met in a large familiarity whose Olympian laughter awed the fearful small boy that flitted uneasily past and chilled the slouching senior that rashly paused to examine the notices in assertion of an unearned right.",
    },
    {
      time: "two minutes to three",
      book: "The Chronicle of Young Satan",
      author: "Mark Twain",
      prefix:
        "We betted that it would happen on the morrow; they took us up and gave us the odds of two to one; we betted that it would happen in the afternoon; we got odds of four to one on that; we betted that it would happen at ",
      suffix: "; they willingly granted us the odds of ten to one on that.",
    },
  ],
  "15:00": [
    {
      time: "three o'clock",
      book: "A Confederacy of Dunces",
      author: "John Kennedy Toole",
      prefix: "'I gotta get uptown by ",
      suffix: ".'",
    },
    {
      time: "three o'clock",
      book: "Swallows and Amazons",
      author: "Arthur Ransome",
      prefix: '"Remember," they shouted, "battle at ',
      suffix: " sharp. There's no time to lose.\"",
    },
    {
      time: "Three",
      book: "Mrs Dalloway",
      author: "Virginia Woolf",
      prefix:
        "And the sound of the bell flooded the room with its melancholy wave; which receded, and gathered itself together to fall once more, when she heard, distractedly, something fumbling, something scratching at the door. Who at this hour? ",
      suffix: ", good Heavens! Three already!",
    },
    {
      time: "three o'clock",
      book: "Bel-Ami",
      author: "Guy de Maupassant",
      prefix: "At ",
      suffix:
        " on the afternoon of that same day, he called on her. She held out her two hands, smiling in her usual charming, friendly way; and for a few seconds they looked deep into each other's eyes.",
    },
    {
      time: "three o’clock",
      book: "A Scandal in Bohemia",
      author: "Sir Arthur Conan Doyle",
      prefix: "At ",
      suffix:
        " precisely I was at Baker Street, but Holmes had not yet returned.",
    },
    {
      time: "At three",
      book: "The Moonstone",
      author: "Wilkie Collins",
      prefix: "",
      suffix:
        " on the Wednesday afternoon, that bit of the painting was completed",
    },
    {
      time: "at three",
      book: "Essays in Love",
      author: "Alain de Botton",
      prefix:
        "Ditched by the woman I loved, I exalted my suffering into a sign of greatness (lying collapsed on a bed ",
      suffix:
        " in the afternoon), and hence protected myself from experiencing my grief as the outcome of what was at best a mundane romantic break-up. Chloe's departure may have killed me, but it had at least left me in glorious possession of the moral high ground. I was a martyr.",
    },
    {
      time: "three o'clock",
      book: "Sunset Park",
      author: "Paul Auster",
      prefix: "He walks into the Hospital for Broken Things at ",
      suffix:
        " on Monday afternoon. That was the arrangement. If he came in after six o'clock, he was to head straight for the house in Sunset Park.",
    },
    {
      time: "three o’clock",
      book: "Achates McNeil",
      author: "T.C. Boyle",
      prefix: "I had a ",
      suffix:
        " class in psychology, the first meeting of the semester, and I suspected I was going to miss it. I was right. Victoria made a real ritual of the whole thing, clothes coming off with the masturbatory dalliance of a strip show, the covers rolling back periodically to show this patch of flesh or that, strategically revealed.",
    },
    {
      time: "three o'clock",
      book: "Middlemarch",
      author: "George Eliot",
      prefix: "It was ",
      suffix:
        " in the beautiful breezy autumn day when Mr. Casaubon drove off to his Rectory at Lowick, only five miles from Tipton; and Dorothea, who had on her bonnet and shawl, hurried along the shrubbery and across the park that she might wander through the bordering wood with no other visible companionship than that of Monk, the Great St. Bernard dog, who always took care of the young ladies in their walks",
    },
    {
      time: "three-o’clock",
      book: "To kill a mockingbird",
      author: "Harper Lee",
      prefix: "Ladies bathed before noon, after their ",
      suffix:
        " naps, and by nightfall were like soft teacakes with frostings of sweat and sweet talcum.",
    },
    {
      time: "three o'clock",
      book: "Les Miserables",
      author: "Victor Hugo",
      prefix: "M. Madeleine usually came at ",
      suffix: ", and as punctuality was kindness, he was punctual.",
    },
    {
      time: "three o'clock",
      book: "Madame Bovary",
      author: "Gustave Flaubert",
      prefix: "On Wednesday at ",
      suffix:
        ", Monsieur and Madame Bovary, seated in their dog-cart, set out for Vaubyessard, with a great trunk strapped on behind and a bonnet-box in front of the apron. Besides these Charles held a bandbox between his knees.",
    },
    {
      time: "at three",
      book: "Casino Royale",
      author: "Ian Fleming",
      prefix: "The scent and smoke and sweat of a casino are nauseating ",
      suffix:
        " in the morning. Then the soul-erosion produced by high gambling - a compost of greed and fear and nervous tension - becomes unbearable and the senses awake and revolt from it.",
    },
    {
      time: "Three o'clock",
      book: "Nausea",
      author: "Jean-Paul Sartre",
      prefix: "",
      suffix: " is always too late or too early for anything you want to do",
    },
    {
      time: "Three o'clock",
      book: "Cham",
      author: "Jonathan Trigell",
      prefix: "",
      suffix:
        " is the perfect time in Cham, because anything is possible. You can still ski, but also respectably start drinking, the shops have just reopened, the sun is still up. Three o'clock is never too late or too early",
    },
    {
      time: "three o'clock",
      book: "Deaf Sentence",
      author: "David Lodge",
      prefix:
        "Today was the day Alex had appointed for her 'punishment'. I became increasingly nervous as the hour of ",
      suffix:
        " approached. I was alone in the house, and paced restlessly from room to room, glancing at the clocks in each of them.",
    },
  ],
  "15:01": [
    {
      time: "about three",
      book: "A Connecticut Yankee in King Arthur's Court",
      author: "Mark Twain",
      prefix: "The sun was now setting. It was ",
      suffix:
        " in the afternoon when Alisande had begun to tell me who the cowboys were; so she had made pretty good progress with it - for her. She would arrive some time or other, no doubt, but she was not a person who could be hurried.",
    },
  ],
  "15:03": [
    {
      time: "3.03pm.",
      book: "Harare North",
      author: "Brian Chikwava",
      prefix: "I check Shingi's mobile phone - it says it's ",
      suffix:
        " I get out of bed, open my suitcase to take clean socks out and the smell of Mother hit my nose and make me feel dizzy.",
    },
  ],
  "15:04": [
    {
      time: "1504",
      book: "101 Reykjavik",
      author: "Hallgrímur Helgason",
      prefix: "Woken at ",
      suffix: " by Michelangelo hammering away with his chisel.",
    },
  ],
  "15:05": [
    {
      time: "five minutes past three",
      book: "In Cold Blood",
      author: "Truman Capote",
      prefix: "Ultimately, at ",
      suffix:
        ' that afternoon, Smith admitted the falsity of the Fort Scott tale. "That was only something Dick told his family. So he could stay out overnight. Do some drinking."',
    },
  ],
  "15:07": [
    {
      time: "seven minutes past three",
      book: "Twenty Thousand Streets Under The Sky",
      author: "Patrick Hamilton",
      prefix: "The next day was grey, threatening rain. He was there at ",
      suffix:
        ". The clock on the church over the way pointed to it. They had arranged to be there at three fifteen. Therefore, if she had been there when he came, she would have been eight minutes before her time.",
    },
  ],
  "15:08": [
    {
      time: "3 hr 8 m p.m.",
      book: "Ulysses",
      author: "James Joyce",
      prefix:
        "A private wireless telegraph which would transmit by dot and dash system the result of a national equine handicap (flat or steeplechase) of 1 or more miles and furlongs won by an outsider at odds of 50 to 1 at ",
      suffix:
        " at Ascot (Greenwich time), the message being received and available for betting purposes in Dublin at 2.59 p.m.",
    },
  ],
  "15:09": [
    {
      time: "3.09",
      book: "The Pit-Prop Syndicate",
      author: "Freeman Wills Crofts",
      prefix:
        "On the next day he boarded the London train which reaches Hull at ",
      suffix:
        ". At Paragon Station he soon singled out Beamish from Merriman's description.",
    },
  ],
  "15:10": [
    {
      time: "3.10pm",
      book: "The Purple Cloud",
      author: "M.P. Shiel",
      prefix:
        "This time it was only the simple fact that the hands chanced to point to ",
      suffix:
        ", the precise moment at which all the clocks of London had stopped.",
    },
  ],
  "15:13": [
    {
      time: "thirteen minutes past three",
      book: "Virtual Assassin",
      author: "Simon Kearns",
      prefix: "The lift moved. It was ",
      suffix:
        ". The bell gave out its ping. Two men stepped out of the lift, Alan Norman and another man. Tony Blair walked into the office.",
    },
  ],
  "15:14": [
    {
      time: "3.14",
      book: "The Railway Children",
      author: "Edith Nesbit",
      prefix: "A signal sounded. \"There's the ",
      suffix:
        " up,\" said Perks. \"You lie low till she's through, and then we'll go up along to my place, and see if there's any of them strawberries ripe what I told you about.\"",
    },
    {
      time: "THREE fourteen",
      book: "On the Road",
      author: "Jack Kerouac",
      prefix: "I shall be back at exactly ",
      suffix: ", for our hour of revery together, real sweet revery darling",
    },
  ],
  "15:15": [
    {
      time: "quarter past three",
      book: "Keep the Aspidistra Flying",
      author: "George Orwell",
      prefix:
        "Gordon was alone. He wandered back to the door. The strawberry-nosed man glanced over his shoulder, caught Gordon's eye, and moved off, foiled. He had been on the point of slipping Edgar Wallace into his pocket. The clock over the Prince of Wales struck a ",
      suffix: ".",
    },
    {
      time: "3:15",
      book: "Where I'm Calling From",
      author: "Raymond Carver",
      prefix:
        'I got out my old clothes. I put wool socks over my regular socks and took my time lacing up the boots. I made a couple of tuna sandwiches and some double-decker peanut-butter crackers. I filled my canteen and attached the hunting knife and the canteen to my belt. As I was going out the door, I decided to leave a note. So I wrote: "Feeling better and going to Birch Creek. Back soon. R. ',
      suffix: '." That was about four hours from now.',
    },
    {
      time: "3:15",
      book: "The Voices of Time",
      author: "JG Ballard",
      prefix:
        "July 3: 5 3/4 hours. Little done today. Deepening lethargy, dragged myself over to the lab, nearly left the road twice. Concentrated enough to feed the zoo and get the log up to date. Read through the operating manuals Whitby left for the last time, decided on a delivery rate of 40 rontgens/min., target distance of 530 cm. Everything is ready now. Woke 11:05. To sleep ",
      suffix: ".",
    },
  ],
  "15:16": [
    {
      time: "1516",
      book: "The Crow Road",
      author: "Iain Banks",
      prefix: "The Nimrod rendezvoused with the light aircraft at ",
      suffix: " GMT.",
    },
  ],
  "15:20": [
    {
      time: "twenty minutes past three",
      book: "Occupied City",
      author: "David Peace",
      prefix: "At ",
      suffix:
        " on Monday, 26 January 1948, in Tokyo, and I am drinking and I am drinking and I am drinking and I am drinking and I am drinking and I am drinking and I am drinking and I am drinking and I am drinking …",
    },
  ],
  "15:23": [
    {
      time: "Three twenty-three",
      book: "Espedair Street",
      author: "Iain Banks",
      prefix: "",
      suffix:
        "! Is that all? Doesn't time - no, I've already said that, thought that. I sit and watch the seconds change on the watch. I used to have a limited edition Rolex worth the price of a new car but I lost it",
    },
    {
      time: "Three twenty-three",
      book: "Espedair Street",
      author: "Iain Banks",
      prefix: "",
      suffix:
        "! Is that all? Doesn't time - no, I've already said that, thought that. I sit and watch the seconds change on the watch. I used to have a limited edition Rolex worth the price of a new car but I lost it. It was present from...Christine? No, Inez. She got fed up with me always having to ask other people what the time was; embarrassed on my behalf",
    },
  ],
  "15:25": [
    {
      time: "15.25",
      book: "C",
      author: "Tom McCarthy",
      prefix:
        "\"Hmm, let's see. It's a three-line rail-fence, a, d, g...d-a-r-l...Got it: 'Darling Hepzibah'—Hepzibah? What kind of name is that?—'Will meet you Reading Sunday ",
      suffix: " train Didcot-Reading.' Reading you all right, you idiots.\"",
    },
  ],
  "15:27": [
    {
      time: "3.27pm",
      book: "The Curious Incident Of The Dog In The Night-Time",
      author: "Mark Haddon",
      prefix: "And she rang the Reverend Peters and he came into school at ",
      suffix: " and he said, 'So, young man, are we ready to roll?'",
    },
  ],
  "15:29": [
    {
      time: "nearly half-past three",
      book: "Nuns at Luncheon",
      author: "Aldous Huxley",
      prefix: '"Good heavens!" she said, "it\'s ',
      suffix:
        ". I must fly. Don't forget about the funeral service,\" she added, as she put on her coat. \"The tapers, the black coffin in the middle of the aisle, the nuns in their white-winged coifs, the gloomy chanting, and the poor cowering creature without any teeth, her face all caved in like an old woman's, wondering whether she wasn't really and in fact dead - wondering whether she wasn't already in hell. Goodbye.\"",
    },
  ],
  "15:30": [
    {
      time: "half-past thrrree",
      book: "The Witches",
      author: "Roald Dahl",
      prefix:
        "\"Before I am rrroasting the alarm-clock, I am setting it to go off, not at nine o'clock the next morning, but at ",
      suffix:
        ' the next afternoon. Vhich means half-past thrrree this afternoon. And that", she said, glancing at her wrist-watch, "is in prrree-cisely seven minutes\' time!"',
    },
    {
      time: "3.30 p.m.",
      book: "The Curious Incident of the Dog in the Night-time",
      author: "Mark Haddon",
      prefix: "",
      suffix: " Catch school bus hom",
    },
    {
      time: "half past three",
      book: "When We Were Orphans",
      author: "Kazuo Ishiguro",
      prefix:
        "I must have completed my packing with time to spare, for when the knock came on my door at ",
      suffix:
        " precisely, I had been sitting in my chair waiting for a good while. I opened the door to a young Chinese man, perhaps not even twenty, dressed in a gown, his hat in his hand.",
    },
  ],
  "15:32": [
    {
      time: "3:32",
      book: "The Fault in Our Stars",
      author: "John Green",
      prefix: "At ",
      suffix:
        " precisely, I noticed Kaitlyn striding confidently past the Wok House. She saw me the moment I raised my hand, flashed her very white and newly straightened teeth at me, and headed over.",
    },
  ],
  "15:33": [
    {
      time: "Three thirty-three",
      book: "11/22/63",
      author: "Stephen King",
      prefix:
        "I picked up my briefcase, glancing at my watch again as I did so. ",
      suffix: ".",
    },
  ],
  "15:35": [
    {
      time: "three-thirty-five",
      book: "Anagrams",
      author: "Lorrie Moore",
      prefix: "By ",
      suffix:
        " business really winds down. I have already sold my ladderback chairs and my Scottish cardigans. I'm not even sure now why I've sold all these things, except perhaps so as not to be left out of this giant insult to one's life that is a yard sale, this general project of getting rid quick.",
    },
    {
      time: "3:35 P.M.",
      book: "I Am No One You Know: Stories",
      author: "Joyce Carol Oates",
      prefix:
        "If Me flashed a little crazy after a restless night of smoking & prowling the darkened house with owl-eyes alert to suspicious noises outside & on the roof, it didn’t inevitably mean she’d still be in such a state when the schoolbus deposited Wolfie back home at ",
      suffix: "",
    },
  ],
  "15:37": [
    {
      time: "15.37",
      book: "The Long Dark Tea Time of the Soul",
      author: "Douglas Adams",
      prefix:
        'The explosion was now officially designated an "Act of God". But, thought Dirk, what god? And why? What god would be hanging around Terminal Two of Heathrow Airport trying to catch the ',
      suffix: " flight to Oslo?",
    },
  ],
  "15:39": [
    {
      time: "three thirty-nine",
      book: "11/22/63",
      author: "Stephen King",
      prefix:
        "I lived two lives in late 1965 and early 1963, one in Dallas and one in Jodie. They came together at ",
      suffix: " in the afternoon of April 10.",
    },
  ],
  "15:40": [
    {
      time: "three-forty",
      book: "Watchers",
      author: "Dean Koontz",
      prefix: "At ",
      suffix:
        ", Cliff called to report that Dilworth and his lady friend were sitting on the deck of the Amazing Grace, eating fruit and sipping wine, reminiscing a lot, laughing a little. “From what we can pick up with directional microphones and from what we can see, I’d say they don’t have any intention of going anywhere. Except maybe to bed. They sure do seem to be a randy old pair.” “Stay with them,” Lem said. “I don’t trust him.”",
    },
  ],
  "15:41": [
    {
      time: "15:41",
      book: "The Crow Road",
      author: "Iain Banks",
      prefix: "At ",
      suffix:
        " GMT, the Cessna's engine began to cut out and the plane - presumably out of fuel - began to lose altitude",
    },
  ],
  "15:44": [
    {
      time: "3.44 p.m.",
      book: "The Girl who Played with Fire",
      author: "Stieg Larsson",
      prefix:
        "The armed response team hastily assembled from Strängnäs arrived at Bjurman's summer cabin at ",
      suffix: "",
    },
  ],
  "15:45": [
    {
      time: "3.45pm",
      book: "11/22/63",
      author: "Stephen King",
      prefix:
        'I opened my notebook, flipped almost to the end before I found a blank page, and wrote "October 5th, ',
      suffix:
        ", Dunning to Longview Cem, puts flowers on parents' (?) graves. Rain.\" I had what I wanted.",
    },
    {
      time: "3:45",
      book: "The Voices of Time",
      author: "JG Ballard",
      prefix:
        "One meal is enough now, topped up with a glucose shot. Sleep is still 'black', completely unrefreshing. Last night I took a 16 mm. film of the first three hours, screened it this morning at the lab. The first true-horror movie. I looked like a half-animated corpse. Woke 10:25. To sleep ",
      suffix: ".",
    },
  ],
  "15:49": [
    {
      time: "3.49 p.m.",
      book: "The Curious Incident of the Dog in the Night-time",
      author: "Mark Haddon",
      prefix: "",
      suffix: " Get off school bus at hom",
    },
    {
      time: "3.49 pm",
      book: "The Curious Incident of the Dog in the Night-time",
      author: "Mark Haddon",
      prefix:
        "But there were more bad things than good things. And one of them was that Mother didn't get back from work til 5.30 pm so I had to go to Father's house between ",
      suffix:
        " and 5.30 pm because I wasn't allowed to be on my own and Mother said I didn't have a choice so I pushed the bed against the door in case Father tried to come in.",
    },
  ],
  "15:50": [
    {
      time: "3.50 p.m.",
      book: "The Curious Incident Of The Dog In The Night-Time",
      author: "Mark Haddon",
      prefix: "",
      suffix: " Have juice and snac",
    },
  ],
  "15:51": [
    {
      time: "fifty-one minutes after fifteen o'clock",
      book: "Italian Without a Master",
      author: "Mark Twain",
      prefix:
        "Date of the telegram, Rome, November 24, ten minutes before twenty-three o'clock. The telegram seems to say, \"The Sovereigns and the Royal Children expect themselves at Rome tomorrow at ",
      suffix: '."',
    },
  ],
  "15:53": [
    {
      time: "Seven minutes to four",
      book: "Tripwire",
      author: "Lee Child",
      prefix:
        "It was like the clouds lifting away from the sun. Jodie glanced at Reacher. He glanced at the clock. ",
      suffix: ". Less than three hours to go.",
    },
  ],
  "15:55": [
    {
      time: "3.55 p.m.",
      book: "The Curious Incident of the Dog in the Night-time",
      author: "Mark Haddon",
      prefix: "",
      suffix: " Give Toby food and wate",
    },
  ],
  "15:56": [
    {
      time: "Four minutes to four",
      book: "Tripwire",
      author: "Lee Child",
      prefix: "",
      suffix: ". Newman sighed again, lost in thought",
    },
  ],
  "15:57": [
    {
      time: "close upon four",
      book: "The Adventures of Sherlock Holmes",
      author: "Arthur Conan Doyle",
      prefix: "It was ",
      suffix:
        " before the door opened, and a drunken-looking groom, ill-kempt and side-whiskered with an inflamed face and disreputable clothes, walked into the room. Accustomed as I was to my friend's amazing powers in the use of disguises, I had to look three times before I was certain that it was indeed he.",
    },
  ],
  "15:58": [
    {
      time: "Towards four o'clock",
      book: "Les Miserables",
      author: "Victor Hugo",
      prefix: "",
      suffix:
        ' the condition of the English army was serious. The Prince of Orange was in command of the centre, Hill of the right wing, Picton of the left wing. The Prince of Orange, desperate and intrepid, shouted to the Hollando-Belgians: "Nassau! Brunswick! Never retreat!',
    },
  ],
  "15:59": [
    {
      time: "nearly 4",
      book: "The Blue Afternoon",
      author: "William Boyd",
      prefix: "He looked at his watch: it was ",
      suffix:
        ". He helped Delphine to her feet and led her down a passage to a rear door that gave on to the hospital garden.",
    },
  ],
  "16:00": [
    {
      time: "four o'clock",
      book: "Sense and Sensibility",
      author: "Jane Austen",
      prefix: "... when they all sat down to table at ",
      suffix:
        ", about three hours after his arrival, he had secured his lady, engaged her mother's consent, and was not only in the rapturous profession of the lover, but, in the reality of reason and truth, one of the happiest of men.",
    },
    {
      time: "at four",
      book: "“Toads Revisited” - The Whitsun Weddings",
      author: "Philip Larkin",
      prefix: '"What else can I answer, When the lights come on ',
      suffix: ' At the end of another year"',
    },
    {
      time: "4.00 p.m.",
      book: "The Curious Incident of the Dog in the Night-time",
      author: "Mark Haddon",
      prefix: "",
      suffix: " Take Toby out of his cag",
    },
    {
      time: "four o'clock",
      book: "The Last Temptation",
      author: "Val McDermid",
      prefix:
        "As he turned off towards the fishing village of Cellardyke, the familiar pips announced the ",
      suffix:
        " news. The comforting voice of the newsreader began the bulletin. 'The convicted serial killer and former TV chat show host Jacko Vance has begun his appeal against conviction.",
    },
    {
      time: "at four",
      book: "Memento Mori",
      author: "Muriel Spark",
      prefix: "Charmian woke ",
      suffix: " and sensed the emptiness of the house.",
    },
    {
      time: "at four",
      book: "Atomised",
      author: "Michel Houellebecq",
      prefix: "Djerzinski arrived punctually ",
      suffix:
        " o’clock. Desplechin had asked to see him. The case was intriguing. Certainly, it was common for a researcher to take a year’s sabbatical to work in Norway or Japan, or one of those sinister countries where middle aged people committed suicide en masse.",
    },
    {
      time: "Four o'clock",
      book: "Sad Steps",
      author: "Philip Larkin",
      prefix: "",
      suffix: ": wedge-shaped gardens lie Under a cavernous, a wind-picked sky",
    },
    {
      time: "Four o'clock",
      book: "The Act of Love",
      author: "Howard Jacobson",
      prefix: "",
      suffix:
        ": when time in the city quivers on its axis - the day not yet spent, the wheels of evening just beginning to turn. The handover hour, was how Marius liked to think of it",
    },
    {
      time: "Four o’clock",
      book: "The Woman in White - The Story Continued",
      author: "Walter Hartwright VII",
      prefix: "",
      suffix:
        " has just struck. Good! Arrangement, revision, reading from four to five. Short snooze of restoration for myself, from five to six. Affair of agent and sealed letter from seven to eight. At eight, en route",
    },
    {
      time: "Four o’clock",
      book: "The Woman in White - The Story Continued",
      author: "Wilkie Collins",
      prefix: "",
      suffix:
        " has just struck. Good! Arrangement, revision, reading from four to five. Short snooze of restoration for myself, from five to six. Affair of agent and sealed letter from seven to eight. At eight, en route",
    },
    {
      time: "four o'clock",
      book: "The Cellist Of Sarajevo",
      author: "Steven Galloway",
      prefix:
        "He played for twenty-two days, just as he said he would. Every day at ",
      suffix:
        " in the afternoon, regardless of how much fighting was going on around him.",
    },
    {
      time: "four o’clock",
      book: "Blood Bride",
      author: "Susan May Gudge",
      prefix:
        "Her eyes caught the kryptonite glow of the digital clock on the front of the microwave. Honest and true, the numbers spelled out the time although she, for a moment, found its calculation to be somehow erroneous. It was ",
      suffix: " in the afternoon.",
    },
    {
      time: "four o'clock",
      book: "The Last Chronicle of Barset",
      author: "Anthony Trollope",
      prefix:
        "I doubt whether anyone was commissioned to send the news along the actual telegraph, and yet Mrs. Proudie knew it before ",
      suffix:
        ". But she did not know it quite accurately.'Bishop', she said, standing at her husband's study door. 'They have committed that man to gaol. There was no help for them unless they had forsworn themselves.'",
    },
    {
      time: "Four O’clock",
      book: "Ghost Generations",
      author: "Susan May Gudge",
      prefix: "I only found out much later that those flowers were called ",
      suffix:
        ", and were not magic at all. The magic was in the seed, waiting to be watered and cared for, the real magic was life.",
    },
    {
      time: "four o’clock",
      book: "Life, the universe and everything",
      author: "Douglas Adams",
      prefix:
        "In the end, it was the Sunday afternoons he couldn’t cope with, and that terrible listlessness that starts to set in about 2.55, when you know you’ve had all the baths you can usefully have that day, that however hard you stare at any given paragraph in the newspaper you will never actually read it, or use the revolutionary new pruning technique it describes, and that as you stare at the clock the hands will move relentlessly on to ",
      suffix: ", and you will enter the long dark teatime of the soul.",
    },
    {
      time: "struck four",
      book: "Brave New World",
      author: "Aldous Huxley",
      prefix:
        "In the four thousand rooms of the Centre the four thousand electric clocks simultaneously ",
      suffix:
        '. Discarnate voices called from the trumpet mouths. "Main Day-shift off duty. Second Day-shift take over. Main Day-shift off …"',
    },
    {
      time: "4 o'clock",
      book: "Deaf Sentence",
      author: "David Lodge",
      prefix:
        "It was my turn to cook the evening meal so I didn't linger in the common room. It was exactly ",
      suffix:
        " as I made my way out of the building, and doors opened behind and before me, discharging salvos of vocal babble and the noise of chair-legs scraping on wooden floors.",
    },
    {
      time: "Four",
      book: "Ulysses",
      author: "James Joyce",
      prefix:
        "Miss Douce took Boylan's coin, struck boldly the cashregister. It clanged. Clock clacked. Fair one of Egypt teased and sorted in the till and hummed and handed coins in change. Look to the west. A clack. For me. —What time is that? asked Blazes Boylan. ",
      suffix:
        "? O'clock. Lenehan, small eyes ahunger on her humming, bust ahumming, tugged Blazes Boylan's elbowsleeve. —Let's hear the time, he said.",
    },
    {
      time: "1600h.",
      book: "Infinite Jest",
      author: "David Foster Wallace",
      prefix:
        "The horrifying R.N. wipes Gately's face off as best she can with her hand and says she'll try to fit him in for a sponge bath before she goes off shift at ",
      suffix: ", at which Gately goes rigid with dread.",
    },
  ],
  "16:01": [
    {
      time: "1601",
      book: "101 Reykjavik",
      author: "Hallgrímur Helgason",
      prefix:
        "Light is coming in through the curtains. Suddenly the digits on the clock radio look like a year. ",
      suffix:
        ". I woke up a bit early, don't have to be born for another 400 years.",
    },
  ],
  "16:02": [
    {
      time: "two minutes after four",
      book: "Southern Ghost",
      author: "Carolyn G Hart",
      prefix:
        "I'd just looked up at the clock, to make sure time wasn't getting away from me, when I heard the shot. It was ",
      suffix: ". I didn't know what to do.",
    },
  ],
  "16:03": [
    {
      time: "16.03",
      book: "What was Lost",
      author: "Catherine O'Flynn",
      prefix: "She read the page carefully and then said, '",
      suffix: " - cat goes to the toilet in front garden.'",
    },
  ],
  "16:04": [
    {
      time: "A little after four o'clock",
      book: "The Private Lives of Pippa Lee",
      author: "Rebecca Miller",
      prefix: "",
      suffix:
        ", Pippa meandered over to Dot's house carrying a bottle of wine she had been keeping in reserve and wondering if she could possibly be pregnant in spite of the vestigial coil still lodged in her uterus like astronaut litter abandoned on the moon",
    },
  ],
  "16:05": [
    {
      time: "Five past four",
      book: "The Bell Jar",
      author: "Sylvia Plath",
      prefix:
        "I had met Irwin on the steps of the Widener Library. I was standing at the top of the long flight, overlooking the red brick buildings that walled the snow-filled quad and preparing to catch the trolley back to the asylum, when a tall young man with a rather ugly and bespectacled, but intelligent face, came up and said, 'Could you please tell me the time?' I glanced at my watch. '",
      suffix: ".'",
    },
    {
      time: "Five past four",
      book: "The Bell Jar",
      author: "Sylvia Plath",
      prefix:
        "I was standing at the top of the long flight, overlooking the red brick buildings that walled the snow-filled quad and preparing to catch the trolley back to the asylum, when a tall young man with a rather ugly and bespectacled, but intelligent face, came up and said, 'Could you please tell me the time?' I glanced at my watch. '",
      suffix: ".'",
    },
    {
      time: "five minutes past four",
      book: "Lady Audley's Secret",
      author: "Mary Elizabeth Braddon",
      prefix: "IT was exactly ",
      suffix:
        " as Mr. Robert Audley stepped out upon the platform at Shoreditch, and waited placidly … it took a long while to make matters agreeable to all claimants, and even the barrister's seraphic indifference to mundane affairs nearly gave way.",
    },
  ],
  "16:06": [
    {
      time: "six minutes after four",
      book: "Follow Me: A Novel",
      author: "Joanna Scott",
      prefix: "At ",
      suffix:
        ", Benny's Cadillac pulled up in front of Mr. Botelia's store, and Benny's mother stepped out of the car with Penelope, who was gnawing on the tip of an ice cream cone.",
    },
  ],
  "16:07": [
    {
      time: "seven minutes after four",
      book: "Love in the Time of Cholera",
      author: "Gabriel García Márquez",
      prefix:
        "But he released him immediately because the ladder slipped from under his feet and for an instant he was suspended in air and then he realised that he had died without Communion, without time to repent of anything or to say goodbye to anyone, at ",
      suffix: " on Pentecost Sunday.",
    },
  ],
  "16:08": [
    {
      time: "eight minutes after four",
      book: "The Monkey's Raincoat",
      author: "Robert Crais",
      prefix: "It was ",
      suffix:
        ". I still don't have a plan. Maybe the guys in the Nova, maybe they had a plan.",
    },
  ],
  "16:09": [
    {
      time: "nine minutes after four",
      book: "Rosemary's Baby",
      author: "Ira Levin",
      prefix:
        'I have to hang up now, Rosemary said. "I just wanted to know if there was any improvement." "No, there isn\'t. It was nice of you to call." She hung up. It was ',
      suffix: ".",
    },
  ],
  "16:10": [
    {
      time: "1610h.",
      book: "Infinite Jest",
      author: "David Foster Wallace",
      prefix: "",
      suffix:
        " E.T.A Weight room. Freestyle circuits. The clank and click of various resistance systems",
    },
    {
      time: "ten-past four",
      book: "Paula Spencer",
      author: "Roddy Doyle",
      prefix:
        "She looks at the clock. She's in the kitchen. A minute left. She waits. It's ",
      suffix:
        ". She picks up the eclair. She licks the cream out of it. She watches herself.It's fuckin' stupid. But. She bites into the chocolate, and the pastry that's been softened by the cream. Jack's not home yet. Leannes's at work. Paula will be leaving, herself, in a bit. She's a year off the drink. Exactly a year. She looks at the clock. A year and a minute.",
    },
  ],
  "16:11": [
    {
      time: "4:11 P.M.",
      book: "Seek",
      author: "Denis Johnson",
      prefix: "",
      suffix:
        " Thurs. A Huey helicopter flies east overhead as the last of the U.S. Marines make ready to leave the beach; a buzzard dangles in the thermals closer over the town",
    },
  ],
  "16:12": [
    {
      time: "twelve minutes after four",
      book: "The Empty Mirror",
      author: "J Sydney Jones",
      prefix: "At precisely ",
      suffix:
        " a body of cavalry rode into the square, four abreast, clearing a way for the funeral cortege.",
    },
  ],
  "16:13": [
    {
      time: "4.13pm",
      book: "Aunt Julia and the Scriptwriter",
      author: "Mario Vargas Llosa",
      prefix: "But at precisely ",
      suffix:
        ", the fifty thousand spectators saw the totally unexpected happen, before their very eyes. From the most crowded section of the southern grandstand, an apparition suddenly emerged.",
    },
  ],
  "16:14": [
    {
      time: "4.14pm",
      book: "Already Dead",
      author: "Denis Johnson",
      prefix: "Then at ",
      suffix:
        " on March 12 I moved behind zinc-zirconium-not-to-be-revealed-compounds protecting me in this hill, and God have mercy but the struggle is just exchanged for the next one, which is exhausting me further as I say, to separate the true from the false.",
    },
  ],
  "16:15": [
    {
      time: "quarter past four",
      book: "False Security",
      author: "John Betjeman",
      prefix: "I remember the dread with which I at ",
      suffix: "/ Let go with a bang behind me our house front door",
    },
    {
      time: "quarter past four",
      book: "Northanger Abbey",
      author: "Jane Austen",
      prefix: "It is only a ",
      suffix:
        ", (shewing his watch) and you are not now in Bath. No theatre, no rooms to prepare for. Half an hour at Northanger must be enough.",
    },
    {
      time: "4:15",
      book: "The Voices of Time",
      author: "JG Ballard",
      prefix:
        "Must have the phone disconnected. Some contractor keeps calling me up about payment for 50 bags of cement he claims I collected ten days ago. Says he helped me load them onto a truck himself. I did drive Whitby's pick-up into town but only to get some lead screening. What does he think I'd do with all that cement? Just the sort of irritating thing you don't expect to hang over your final exit. (Moral: don't try too hard to forget Eniwetok.) Woke 9:40. To sleep ",
      suffix: ".",
    },
    {
      time: "quarter past four",
      book: "Play it as it Lays",
      author: "Joan Didion",
      prefix: "On the tenth day of October at ",
      suffix:
        " in the afternoon with a dry hot wind blowing through the passed Maria found herself in Baker. She had never meant to go as far as Baker, had started out that day as every day, her only destination the freeway. But she had driven out of San Bernadino and up the Barstow and instead of turning back at Barstow (she had been out that far before but never that late in the day, it was past time to navigate back, she was out too far too late, the rhythm was lost ) she kept driving.",
    },
    {
      time: "4.15",
      book: "The Wind-up Bird Chronicle",
      author: "Haruki Murakami",
      prefix:
        "The sun had begun to sink in the west, and the shadow of an oak branch had crept across my knees. My watch said it was ",
      suffix: ".",
    },
  ],
  "16:16": [
    {
      time: "4.16pm",
      book: "The Winner Stands Alone",
      author: "Paulo Coelho",
      prefix: "",
      suffix:
        " The terrace outside the bar is packed, and Igor feels proud of his ability to plan things, because even though he's never been to Cannes before, he had foreseen precisely this situation and reserved a table",
    },
  ],
  "16:17": [
    {
      time: "four-seventeen",
      book: "A Prefect's Uncle",
      author: "P.G. Wodehouse",
      prefix: "Apparently the great Percy has no sense of humour, for at ",
      suffix:
        " he got tired of it, and hit Skinner crisply in the right eyeball, blacking the same as per illustration.",
    },
    {
      time: "seventeen minutes after four",
      book: "Life Penalty",
      author: "Joy Fielding",
      prefix:
        "In the next instant she was running toward her house, unmindful of the bags she had dropped, seeing only the police cars, knowing as she glanced down at her watch and saw that it was ",
      suffix: ", that for her time had stopped.",
    },
  ],
  "16:18": [
    {
      time: "4.18 p.m.",
      book: "The Curious Incident of the Dog in the Night-time",
      author: "Mark Haddon",
      prefix: "",
      suffix: " Put Toby into his cag",
    },
  ],
  "16:19": [
    {
      time: "4:19 PM",
      book: "The Other Woman",
      author: "Eric Jerome Dickey",
      prefix: "Jessica [",
      suffix:
        "] Don't tease me like that. I haven't been to a play in years. Charles [4:19 PM] Then it'll be my treat. You and the hubby can have big fun on me.",
    },
  ],
  "16:20": [
    {
      time: "4.20 p.m.",
      book: "The Curious Incident of the Dog in the Night-time",
      author: "Mark Haddon",
      prefix: "",
      suffix: " Watch television or a vide",
    },
    {
      time: "twenty minutes past four",
      book: "Raise High the Roof Beam, Carpenters",
      author: "J. D. Salinger",
      prefix: "At ",
      suffix:
        " - or, to put it another, blunter way, an hour and twenty minutes past what seemed to be all reasonable hope - the unmarried bride, her head down, a parent stationed on either side of her, was helped out of the building...",
    },
  ],
  "16:21": [
    {
      time: "4.21pm",
      book: "Hunted Past Reason",
      author: "Richard Matheson",
      prefix: "",
      suffix:
        ' As they started on, Doug picked up a twig and after rubbing it off, started to move one end of it inside his mouth. "What are you doing?" Bob asked. "Brushing my teeth, nature style," Doug answered. Bob grunted, smiling slightly. "I\'ll use my toothbrush," he said',
    },
  ],
  "16:22": [
    {
      time: "4.22pm",
      book: "Balance of Power: Op-Center 05",
      author: "Tom Clancy, Steve Pieczenik, and Jeff Rovin",
      prefix: "Monday, ",
      suffix:
        " Washington, D.C. Paul Hood took his daily late-afternoon look at the list of names on his computer monitor.",
    },
  ],
  "16:23": [
    {
      time: "4.23",
      book: "A Visit from the Goon Squad",
      author: "Jennifer Egan",
      prefix:
        "They were hurrying west, trying to reach the river before sunset. The warming-related 'adjustments' to Earth's orbit had shortened the winter days, so that now, in January, sunset was taking place at ",
      suffix: ".",
    },
  ],
  "16:24": [
    {
      time: "4:24",
      book: "Teardrop",
      author: "Travis Thrasher",
      prefix:
        "Mike winked at Ashley and continued with the remaining greetings and hugs and handshakes. The time was ",
      suffix: ". Six hours to go. The minutes seemed to just melt away.",
    },
  ],
  "16:25": [
    {
      time: "twenty-five minutes past four",
      book: "The Man with the Twisted Lip",
      author: "Arthur Conan Doyle",
      prefix:
        "As I dressed I glanced at my watch. It was no wonder that no one was stirring. It was ",
      suffix:
        ". I had hardly finished when Holmes returned with the news that the boy was putting in the horse.",
    },
  ],
  "16:26": [
    {
      time: "twenty-six minutes after four",
      book: "Catch-As-Catch-Can",
      author: "Charlotte Armstrong",
      prefix:
        "It seemed all wrong to have thought of such a thing. She thought, \"I don't know him. Nor does he know me. Nor ever shall we.” She put her bare hand in the sun, where the wind would weather it. It was ",
      suffix: ".",
    },
  ],
  "16:28": [
    {
      time: "4.28pm",
      book: "The Ruined Map: A Novel",
      author: "Kobo Abe",
      prefix: "Same day: ",
      suffix:
        "- Right turn at the second bus stop after the gas station. I stopped the car at the first ward post office and inquired at the corner tobacconists. Mr. M's house was the one to the right of the post office, visible diagonally in front of me.",
    },
  ],
  "16:29": [
    {
      time: "4:29 pm.",
      book: "Believing Cedric",
      author: "Mark Lavorato",
      prefix: "October 21, 2007, ",
      suffix:
        " The phone was red. And what William hated most about it, besides the fact that it was inconveniently mounted on a wall in a tight corner (and at a strange angle), was that when it rang it was so gratingly loud that you could actually see the cherry receiver quavering as you picked it up.",
    },
  ],
  "16:30": [
    {
      time: "four-thirty",
      book: "Odd Hours",
      author: "Dean Koontz",
      prefix: "At ",
      suffix:
        " that afternoon in late January when I stepped into the parlour with Boo, my dog, Hutch was in his favourite armchair, scowling at the television, which he had muted.",
    },
    {
      time: "four thirty",
      book: "American Psycho",
      author: "Bret Easton Ellis",
      prefix: "I leave the office at ",
      suffix:
        ", head up to Xclusive where I work out on free weights for an hour, then taxi across the park to Gio's in the Pierre Room for a facial, a manicure and, if time permits, a pedicure.",
    },
    {
      time: "four-thirty",
      book: "Essays in Love",
      author: "Alain de Botton",
      prefix:
        "She hung up on me at first, then asked me whether I made a point of behaving like a 'small-time suburban punk' with women I had slept with. But after apologies, insults, laughter, and tears, Romeo and Juliet were to be seen together later that afternoon, mushily holding hands in the dark at a ",
      suffix:
        " screening of L ove and Death at the National Film Theatre. Happy endings – for now at least.",
    },
  ],
  "16:31": [
    {
      time: "4:31 PM",
      book: "Click: An Online Love Story",
      author: "Lisa Becker",
      prefix: "From: Renee Greene – August 5, 2011 – ",
      suffix:
        " To: Shelley Manning Subject: Re: All Access What should I be worried about, then? JUST KIDDING. You're right. Well, I gotta run, my groupie friend. I actually have REAL work to do. I'll talk to you tonight.",
    },
  ],
  "16:32": [
    {
      time: "4.32pm.",
      book: "Seek",
      author: "Denis Johnson",
      prefix: "",
      suffix:
        ' Now the eight Marines next to us leave their emplacement and file quickly past, the last saying, "Go! Go! Go!" They break into a run',
    },
  ],
  "16:33": [
    {
      time: "4.33pm",
      book: "Havana World Series",
      author: "José Latour",
      prefix: "At ",
      suffix:
        ", a short bald man puffing on a cigar arrived at the library. He approached a huge cabinet storing thousands of alphabetically arranged cards and slid a drawer out. The tips of his fingers were bandaged.",
    },
  ],
  "16:34": [
    {
      time: "4.34 p.m.",
      book: "The Raw Shark Texts",
      author: "Steven Hall",
      prefix:
        "A bedroom stocked with all the ordinary, usual things. There was a wardrobe in the corner. A bedside table with a collection of water glasses of varying ages and an alarm clock with red digital numbers- ",
      suffix: "",
    },
  ],
  "16:35": [
    {
      time: "4.35",
      book: "4.50 from Paddington",
      author: "Agatha Christie",
      prefix:
        "The Voice shut itself off with a click, and then reopened conversation by announcing the arrival at Platform 9 of the ",
      suffix: " from Birmingham and Wolverhampton.",
    },
  ],
  "16:37": [
    {
      time: "1637.",
      book: "101 Reykjavik",
      author: "Hallgrímur Helgason",
      prefix: "She should have been home by now. ",
      suffix:
        " Yes. It's as if I had the date of a year on my arm. Every day is a piece of world history.",
    },
  ],
  "16:39": [
    {
      time: "4:39 p.m.",
      book: "Blood Red Blues",
      author: "Teddy Hayes",
      prefix:
        "Harlem enjoys lazy Sabbath mornings, although the pace picks up again in the afternoon, after church. My watch read ",
      suffix:
        ", and I realized that I hadn't eaten all day. I bought two slices of pizza from a sidewalk vendor on 122nd and Lenox Avenue and washed it down with a grape Snapple.",
    },
  ],
  "16:40": [
    {
      time: "Four forty",
      book: "Trouble & Triumph: A Novel of Power & Beauty",
      author: 'Tip "T.I." Harris with David Ritz',
      prefix: "",
      suffix:
        " P.M. Besta sang another hymn. Everyone knew something was wrong. How long did they wait? The mayor was going crazy inside, as was the mayor's wife, as was their daughter. Seiji could barely contain his rage. He was turning as red as his red tuxedo",
    },
  ],
  "16:42": [
    {
      time: "4:42pm.",
      book: "What I Talk About When I Talk About Running",
      author: "Haruki Murakami",
      prefix:
        "I'm always happy when I reach the finish line of a long-distance race, but this time it really struck me hard. I pumped my right fist into the air. The time was ",
      suffix:
        " Eleven hours and forty-two minutes since the start of the race.",
    },
  ],
  "16:45": [
    {
      time: "four-forty-five",
      book: "Co-ordination",
      author: "EM Forster",
      prefix: "At ",
      suffix:
        " Miss Haddon went to tea with the Principal, who explained why she desired all the pupils to learn the same duet. It was part of her new co-ordinative system.",
    },
    {
      time: "fifteen minutes before five",
      book: "Where I'm Calling From",
      author: "Raymond Carver",
      prefix:
        "The next day Bill took only ten minutes of the twenty-minute break allotted for the afternoon and left at ",
      suffix:
        ". He parked the car in the lot just as Arlene hopped down from the bus.",
    },
  ],
  "16:46": [
    {
      time: "4:46",
      book: "The Havana World Series",
      author: "José Latour",
      prefix: "At ",
      suffix:
        " an obese, middle-aged man shuffled in. Wearing a starched guayabera and dark green pants, Ureña asked for a book on confectionery, then took a seat at the end of the same reading room. Evelina and Leticia exchanged astonished glances. It definitely was one of those days.",
    },
  ],
  "16:47": [
    {
      time: "4:47",
      book: "The Art of Fielding",
      author: "Chad Harbach",
      prefix:
        "But maybe it was more than that, maybe Affenlight had erred badly somehow, because here it was 4:49 by his watch, ",
      suffix: " by the wall clock, and Owen had not yet come.",
    },
  ],
  "16:48": [
    {
      time: "4:48 a.m.",
      book: "What is the What",
      author: "Dave Eggers",
      prefix:
        "Thinking about the card warms me to the idea of walking under the arched doorway of the Newtons' home, but when I arrive at their house, the plan seems ridiculous. What am I doing? It's ",
      suffix: ", and I'm parked outside their darkened house.",
    },
  ],
  "16:49": [
    {
      time: "4:49 p.m.",
      book: "Beyond Recognition",
      author: "Ridley Pearson",
      prefix: "",
      suffix:
        ", a bald-headed man wearing khakis and ankle-high deck shoes came out through the front door of the purple house on 21st Avenue East. The detectives had nicknamed him the General",
    },
  ],
  "16:50": [
    {
      time: "4.50",
      book: "4.50 from Paddington",
      author: "Agatha Christie",
      prefix:
        '"The train standing at Platform 3," the Voice told her, "is the ',
      suffix:
        ' for Brackhampton, Milchester, Waverton, Carvil Junction, Roxeter and stations to Chadmouth. Passengers for Brackhampton and Milchester travel at the rear of the train. Passengers for Vanequay change at Roxeter." The voice shut itself off with a click,',
    },
    {
      time: "ten minutes to five",
      book: "The 13 Clocks",
      author: "James Thurber",
      prefix:
        "They had all frozen at the same time, on a snowy night, seven years before, and after that it was always ",
      suffix: " in the castle.",
    },
    {
      time: "ten minutes to five",
      book: "Lamb to the Slaughter",
      author: "Roald Dahl",
      prefix: "When the clock said ",
      suffix:
        ", she began to listen, and a few moments later, punctually as always, she heard the tires on the gravel outside, and the car door slamming, the footsteps passing the window, the key turning in the lock. She laid aside her sewing, stood up, and went forward to kiss him as he came in.",
    },
  ],
  "16:51": [
    {
      time: "Nine minutes to five.",
      book: "Compulsory Happiness",
      author: "Norman Manea and Linda Coverdale",
      prefix: "",
      suffix:
        " If this wasn't some new ordeal, intended to fray her nerves to shreds, if this important person really did exist, if he'd actually set up this appointment, and if, moreover, he arrived on time, then there were nine minutes left",
    },
  ],
  "16:52": [
    {
      time: "eight minutes to five",
      book: "Chaos and Night",
      author: "Henry De Montherlant",
      prefix:
        "The corrida was to begin at five o'clock. The five-footed beasts make a point of arriving at the latest at eight or seven minutes to: ritual again. At ",
      suffix:
        ", there they were. The urchins gave them a tap on the shoulder: another bit of ritual.",
    },
  ],
  "16:53": [
    {
      time: "seven minutes before five",
      book: "The Silence of Bonaventure Arrow",
      author: "Rita Leganski",
      prefix:
        "It was so quiet in the post office that Trinidad could hear the soft tick of the clock's second hand every time it moved. It was now ",
      suffix: ".",
    },
  ],
  "16:54": [
    {
      time: "six minutes before five",
      book: "The Seventeen Widows of Sans Souci",
      author: "Charlotte Armstrong",
      prefix: "At ",
      suffix:
        " o'clock, Daisy Robinson, about to reach her own apartment door, paused to look and to listen. Something was out of order. Tess Rogan's door was standing wide open and, from within, Daisy could hear something being broken.",
    },
    {
      time: "1654",
      book: "The Hunt for Red October",
      author: "Tom Clancy",
      prefix: "It was ",
      suffix:
        " local time when the Red October broke the surface of the Atlantic Ocean for the first time, forty-seven miles southeast of Norfolk. There was no other ship in sight.",
    },
  ],
  "16:55": [
    {
      time: "five minutes to five",
      book: "The Ragged Trousered Philanthropists",
      author: "Robert Tressell",
      prefix: "About ",
      suffix:
        ", just as they were all putting their things away for the night, Nimrod suddenly appeared in the house. He had come hoping to find some of them ready dressed to go home before the proper time.",
    },
  ],
  "16:56": [
    {
      time: "4:56 P.M.",
      book: "Looking for Alaska",
      author: "John Green",
      prefix:
        "And when that final Friday came, when my packing was mostly done, she sat with my dad and me on the living-room couch at ",
      suffix:
        " and patiently awaited the arrival of the Good-bye to Miles Cavalry.",
    },
  ],
  "16:57": [
    {
      time: "nearly five",
      book: "A Single Pebble",
      author: "John Hershey",
      prefix: "It was ",
      suffix:
        " in the evening when the cook came aboard. He did not have the cabbages.",
    },
    {
      time: "three minutes to five",
      book: "The Tailor of Panama",
      author: "John Le Carré",
      prefix: "Then at ",
      suffix:
        " — Pendel had somehow never doubted that Osnard would be punctual — along comes a brown Ford hatchback with an Avis sticker on the back window and pulls into the space reserved for customers.",
    },
  ],
  "16:58": [
    {
      time: "A minute and twenty-one seconds to five",
      book: "The Collected Stories",
      author: "Isaac Bashevis Singer",
      prefix:
        'I was told that in his vest pocket he kept a chronometer instead of a watch. If someone asked him what time it was, he would say, "',
      suffix: '."',
    },
  ],
  "16:59": [
    {
      time: "around 5 p.m.",
      book: "Mortality -- 'The Rainbow'",
      author: "Nicholas Royle",
      prefix: "The rain stopped ",
      suffix:
        " and a few of those people who were out and about expressed mild surprise when the rainbow failed to fade.",
    },
  ],
  "17:00": [
    {
      time: "5.00 p.m.",
      book: "The Curious Incident of the Dog in the Night-time",
      author: "Mark Haddon",
      prefix: "",
      suffix: " Read a boo",
    },
    {
      time: "five",
      book: "The People With the Dogs",
      author: "Christina Stead",
      prefix: "About ",
      suffix:
        ", the Abbot, a young Manchester terrier, began chirruping. He stood on the body of his owner, Flora, with his forepaws on the sill of the balcony, stared through the green rattan blinds, and trembled. He could see the farmer in the field, and Edward asleep on the next balcony.",
    },
    {
      time: "five o'clock",
      book: "The Chateau",
      author: "William Maxwell",
      prefix: "At ",
      suffix:
        " that afternoon, while Barbara waited in a taxi, Harold went into the convent in Auteuil and explained to the nun who sat in the concierge's glass cage that Mme. Straus-Muguet was expecting them. He assumed that men were not permitted any further, and that they would all three go out for tea.",
    },
    {
      time: "five o’clock",
      book: "The Horror at Red Hook",
      author: "H. P. Lovecraft",
      prefix: "At ",
      suffix:
        " adieux were waved, and the ponderous liner edged away from the long pier, slowly turned its nose seaward, discarded its tug, and headed for the widening water spaces that led to old world wonders. By night the outer harbour was cleared, and late passengers watched the stars twinkling above an unpolluted ocean.",
    },
    {
      time: "five o'clock",
      book: "The End of Mr Y",
      author: "Scarlett Thomas",
      prefix: "But I took the mixture at ",
      suffix:
        " in the afternoon. I run my tongue over my dry mouth. I feel dizzy. I know this dizziness: it's because I haven't had a cigarette for hours.",
    },
    {
      time: "five o'clock",
      book: "Jane Eyre",
      author: "Charlotte Brontë",
      prefix: "ERE THE HALF-HOUR ended, ",
      suffix:
        " struck; school was dismissed, and all were gone into the refectory to tea. I now ventured to descend; it was deep dusk; I retired into a corner and sat down on the floor.",
    },
    {
      time: "five o'clock",
      book: "The Portrait of a Lady",
      author: "Henry James",
      prefix: "From ",
      suffix:
        " to eight is on certain occasions a little eternity; but on such an occasion as this the interval could be only an eternity of pleasure.",
    },
    {
      time: "five o'clock",
      book: "Harry Potter and the Philosopher's Stone",
      author: "JK Rowling",
      prefix:
        "He found it harder to concentrate on drills that afternoon and when he left the building at ",
      suffix:
        ", he was still so worried that he walked straight into someone just outside the door.",
    },
    {
      time: "five o'clock",
      book: "The Portrait of a Lady",
      author: "Henry James",
      prefix:
        "She had not seen her yet, as Osmond had given her to understand that it was too soon to begin. She drove at ",
      suffix:
        " to a high floor in a narrow street in the quarter of the Piazza Navona, and was admitted by the portress of the convent, a genial and obsequious person. Isabel had been at this institution before; she had come with Pansy to see the sisters.",
    },
    {
      time: "five o'clock",
      book: "Embers",
      author: "Sandor Marai",
      prefix: "Until ",
      suffix:
        " there was no sign of life from the room. Then he rang for his servant and ordered a cold bath.",
    },
    {
      time: "five o'clock",
      book: "Rebecca",
      author: "Daphne du Maurier",
      prefix:
        "We motored, I remember, leaving London in the morning in a heavy shower of rain, coming to Manderley about ",
      suffix:
        ", in time for tea. I can see myself now, unsuitably dressed as usual, although a bride of seven weeks, in a tan-coloured stockinette frock, a small fur known as a stone marten round my neck, and over all a shapeless mackintosh, far too big for me and dragging to my ankles.",
    },
  ],
  "17:01": [
    {
      time: "One minute after five.",
      book: "Trouble & Triumph: A Novel of Power & Beauty",
      author: 'Tip "T.I." Harris with David Ritz',
      prefix: "",
      suffix:
        " The seated guests were told that the ceremony would begin shortly. A little more patience was required",
    },
  ],
  "17:02": [
    {
      time: "two minutes past five",
      book: "Duplicate Keys",
      author: "Jane Smiley",
      prefix:
        "She stood up, shook her hair into place, smoothed her skirt and turned on the light. It was ",
      suffix: ". She would have thought it midnight or five in the morning.",
    },
  ],
  "17:03": [
    {
      time: "5:03",
      book: "Comrades in Miami: A Novel",
      author: "José Latour",
      prefix:
        '"Good evening, Mrs. Scheindlin," the man said before departing. "Good evening, Chris. Say hello to the wife for me." "I sure will. Thanks. Bye," he said, waving to Elliot, who returned the goodbye. It was ',
      suffix: " when Elliot rested the handset in its cradle.",
    },
  ],
  "17:04": [
    {
      time: "5:04 P.M.",
      book: "The Mothman Prophecies",
      author: "John A. Keel",
      prefix:
        "Frank Wamsley spotted his cousin Barbara and her husband and waved to them. Just ahead, he saw Marvin and his two friends. Suddenly the whole bridge convulsed. The time was ",
      suffix: " Steel screamed.",
    },
  ],
  "17:05": [
    {
      time: "5:05 p.m.",
      book: "Typhoon",
      author: "Charles Cumming",
      prefix: "At approximately ",
      suffix:
        " Joe became aware of a man standing close to the table, about two metres away, talking in Mandarin into a mobile phone. He was a middle-aged Han wearing cheap leather slip-on shoes, high-waisted black trousers and a white short-sleeved shirt.",
    },
  ],
  "17:06": [
    {
      time: "around 5 p.m.",
      book: "Mortality -- 'The Rainbow'",
      author: "Nicholas Royle",
      prefix: "The rain stopped ",
      suffix:
        " and a few of those people who were out and about expressed mild surprise when the rainbow failed to fade.",
    },
  ],
  "17:10": [
    {
      time: "Five ten P.M.",
      book: "Trouble & Triumph: A Novel of Power & Beauty",
      author: 'Tip "T.I." Harris with David Ritz',
      prefix: "",
      suffix:
        " A ground-to-ground cruise missile, launched from a tractor installed in the backyard of Leonard Sudavico's former home by Rashan and a crew of technicians from Afghanistan, exploded onto the Paul Clay estate in the exact spot where the life-size mermaid had once swum in the waterfall",
    },
    {
      time: "ten minutes past five",
      book: "Watchers",
      author: "Dean Koontz",
      prefix: "Hours later, at ",
      suffix:
        ", Saturday afternoon, Nora and Travis and Jim Keene crowded in front of the mattress on which Einstein lay. The dog had just taken a few more ounces of water. He looked at them with interest, too. Travis tried to decide if those large brown eyes still had the strange depth, uncanny alertness, and undoglike awareness that he had seen in them so many times before.",
    },
  ],
  "17:12": [
    {
      time: "twelve minutes past five",
      book: "Rebecca",
      author: "Daphne du Maurier",
      prefix: '"Well, here we are," said Colonel Julyan, "and it\'s exactly ',
      suffix:
        '. We shall catch them in the middle of their tea. Better wait for a bit" Maxim lit a cigarette, and then stretched out his hand to me. He did not speak.',
    },
  ],
  "17:14": [
    {
      time: "fourteen minutes past five",
      book: "To Kill a Mockingbird",
      author: "Harper Lee",
      prefix: '"Do you know what time it is, Atticus?" she said. "Exactly ',
      suffix:
        ". The alarm clock's set for five-thirty. I want you to know that.\"",
    },
  ],
  "17:15": [
    {
      time: "17:15 hrs",
      book: "Bomber",
      author: "Len Deighton",
      prefix:
        "When August Bach emerged from the gloomy chill of the air-conditioned Divisional Fighter Control bunker it was ",
      suffix:
        " CET. The day had ripened into one of those mellow summer afternoons when the air is warm and sweet like soft toffee",
    },
  ],
  "17:18": [
    {
      time: "eighteen minutes past five",
      book: "The Confessions of Arsène Lupin",
      author: "Maurice LeBlanc",
      prefix:
        "Lupin rose, without breaking his contemptuous silence, and took the sheet of paper. I remembered soon after that, at this moment, I happened to look at the clock. It was ",
      suffix: ".",
    },
  ],
  "17:19": [
    {
      time: "5.19 p.m.",
      book: "The Dogs of Riga",
      author: "Henning Mankell",
      prefix: "The call came at ",
      suffix:
        " The line was surprisingly clear. A man introduced himself as Major Liepa from the Riga police. Wallander made notes as he listened, occasionally answering a question.",
    },
  ],
  "17:20": [
    {
      time: "1720",
      book: "Infinite Jest",
      author: "David Foster Wallace",
      prefix:
        "The Meeting was listed as starting at 1730, and it was only around ",
      suffix:
        ", and Hal thought the voices might signify some sort of pre-Meeting orientation for people who've come for the first time, sort of tentatively, just to scout the whole enterprise out, so he doesn't knock.",
    },
  ],
  "17:21": [
    {
      time: "around 1720",
      book: "Infinite Jest",
      author: "David Foster Wallace",
      prefix: "The Meeting was listed as starting at 1730, and it was only ",
      suffix:
        ", and Hal thought the voices might signify some sort of pre-Meeting orientation for people who've come for the first time, sort of tentatively, just to scout the whole enterprise out, so he doesn't knock.",
    },
  ],
  "17:23": [
    {
      time: "Five twenty-three",
      book: "A Fraction of the Whole",
      author: "Steve Toltz",
      prefix:
        '"I was wondering if we could meet for a drink." "What for?" "Just for a chat. Do you know the Royal batsman, near Central Station? We could meet tomorrow at five?" "',
      suffix: '," I said, to exert some control over the situation.',
    },
  ],
  "17:25": [
    {
      time: "five-twenty-five",
      book: "Hard Boiled Wonderland and the End of the World",
      author: "Haruki Murakami",
      prefix: "It was ",
      suffix:
        " when I pulled up in front of the library. Still early for our date, so I got out of the car and took a stroll down the misty streets. In a coffee shop, watched a golf match on television, then I went to an entertainment center and played a video game. The object of the game was to wipe out tanks invading from across the river. I was winning at first, but as the game went on, the enemy tanks bred like lemmings, crushing me by sheer number and destroying my base. An on-screen nuclear blast took care of everything, followed by the message game over insert coin.",
    },
    {
      time: "twenty-five minutes past five",
      book: "A Man Lay Dead",
      author: "Ngaio Marsh",
      prefix:
        "Now said Handsley, when Angela had poured out the last cup, \"it's ",
      suffix: ', At half-past the Murder game is on"',
    },
  ],
  "17:30": [
    {
      time: "half-past five",
      book: "Anna Karenina",
      author: "Leo Tolstoy",
      prefix:
        "He went up to his coachman, who was dozing on the box in the shadow, already lengthening, of a thick lime-tree; he admired the shifting clouds of midges circling over the hot horses, and, waking the coachman, he jumped into the carriage, and told him to drive to Bryansky’s. It was only after driving nearly five miles that he had sufficiently recovered himself to look at his watch, and realise that it was ",
      suffix: ", and he was late.",
    },
    {
      time: "half-past five",
      book: "The Sign Of Four",
      author: "Arthur Conan Doyle",
      prefix: "It was ",
      suffix:
        " before Holmes returned. He was bright, eager, and in excellent spirits, a mood which in his case alternated with fits of the blackest depression.",
    },
  ],
  "17:33": [
    {
      time: "5:33 p.m.",
      book: "Varieties of Disturbance",
      author: "Lydia Davis",
      prefix: "At ",
      suffix:
        " there is a blast of two deep, resonant notes a major third apart. On another day there is the same blast at 12:54 p.m. On another, exactly 8:00 a.m.",
    },
  ],
  "17:37": [
    {
      time: "5:37",
      book: "Lightning Rods",
      author: "Helen DeWitt",
      prefix:
        "Look, Lucille, said Joe when Lucille strolled into the office at ",
      suffix:
        ". \"I don't know what you said to this gal, but it seems to have had exactly the opposite of the desired effect. She's got some bee in her bonnet about Harvard Law School.\"",
    },
  ],
  "17:40": [
    {
      time: "5:40",
      book: "I Love Dollars",
      author: "Zhu Wen",
      prefix:
        "Hey, young man, what time is it? 'What?' I said, is it 5:30 yet? 'Er, ",
      suffix:
        ".' Heavens, they'll be starving. But then that's a good thing. Let them.'",
    },
    {
      time: "five-forty",
      book: "The Undomestic Goddess",
      author: "Sophie Kinsella",
      prefix: "It's ",
      suffix:
        " now. The party's at six. By about ten past, the eleventh floor should be clearing. Arnold is a very popular partner; no one's going to miss his farewell speech if they can help it. Plus, at Carter Spink parties, the speeches always happen early on, so people can get back to work if they need to. And while everyone's listening I'll slip down to Arnold's office. It should work. It has to work. As I stare at my own bizarre reflection, I feel a grim resolve hardening inside me. He's not going to get away with everyone thinking he's a cheery, harmless old teddy bear. He's not going to get away with it.",
    },
  ],
  "17:42": [
    {
      time: "around 5.45",
      book: "Rabbit Is Rich",
      author: "John Updike",
      prefix:
        "Janice is not waiting for him in the lounge or beside the pool when at last ",
      suffix:
        " they come home from playing the par-5 eighteenth. Instead one of the girls in their green and white uniforms comes over and tells him that his wife wants him to call home.",
    },
  ],
  "17:45": [
    {
      time: "5.45",
      book: "Rabbit Is Rich",
      author: "John Updike",
      prefix:
        "Janice is not waiting for him in the lounge or beside the pool when at last around ",
      suffix:
        " they come home from playing the par-5 eighteenth. Instead one of the girls in their green and white uniforms comes over and tells him that his wife wants him to call home.",
    },
  ],
  "17:46": [
    {
      time: "fourteen minutes to six",
      book: "Three Men and a Maid",
      author: "P.G. Wodehouse",
      prefix:
        "Through the curtained windows of the furnished apartment which Mrs. Horace Hignett had rented for her stay in New York rays of golden sunlight peeped in like the foremost spies of some advancing army. It was a fine summer morning. The hands of the Dutch clock in the hall pointed to thirteen minutes past nine; those of the ormolu clock in the sitting-room to eleven minutes past ten; those of the carriage clock on the bookshelf to ",
      suffix:
        ". In other words, it was exactly eight; and Mrs. Hignett acknowledged the fact by moving her head on the pillow, opening her eyes, and sitting up in bed. She always woke at eight precisely.",
    },
  ],
  "17:48": [
    {
      time: "5:48 p.m.",
      book: "The Curious Incident Of The Dog In The Night-Time",
      author: "Mark Haddon",
      prefix: "Father came home at ",
      suffix:
        " I heard him come through the front door. Then he came into the living room. He was wearing a lime green and sky blue check shirt and there was a double knot on one of his shoes but not on the other.",
    },
  ],
  "17:50": [
    {
      time: "Ten to six",
      book: "Noughts and Crosses",
      author: "Malorie Blackman",
      prefix: '"What time is it Jack?" "',
      suffix:
        '""Ten more minutes then." I shuffle the cards. "Time for a quick game of rummy?"',
    },
  ],
  "17:53": [
    {
      time: "Seven minutes to six",
      book: "The Adventures of a Three-Guinea Watch",
      author: "Talbot Baines Reed",
      prefix:
        '"That boy will be spoiled, as sure as I go on springs; he\'s made such a lot of. Have you been regulated?" "I should think I have!" exclaimed I, in indignant recollection of my education. "All right; keep your temper. What time are you?" "',
      suffix: '."',
    },
  ],
  "17:54": [
    {
      time: "5:54 pm",
      book: "The Curious Incident of the Dog in the Night-time",
      author: "Mark Haddon",
      prefix: "It was ",
      suffix:
        " when Father came back into the living room. He said, 'What is this?\" but he said it very quietly and I didn't realise that he was angry because he wasn't shouting.",
    },
  ],
  "17:55": [
    {
      time: "five minutes to six",
      book: "The Deferred Appointment",
      author: "Algernon Blackwood",
      prefix:
        "The wind moaned and sang dismally, catching the ears and lifting the shabby coat-tails of Mr Mortimer Jenkyn, 'Photographic Artist', as he stood outside and put the shutters up with this own cold hands in despair of further trade. It was ",
      suffix: ".",
    },
  ],
  "17:57": [
    {
      time: "nearly six o'clock",
      book: "Dracula",
      author: "Bram Stoker",
      prefix: "When he arrived it was ",
      suffix:
        ", and the sun was setting full and warm, and the red light streamed in through the window and gave more colour to the pale cheeks.",
    },
  ],
  "17:58": [
    {
      time: "nearly six o'clock",
      book: "Burmese Days",
      author: "George Orwell",
      prefix: "It was ",
      suffix:
        " in the evening, and the absurd bell in the six-foot tin steeple of the church went clank-clank, clank- clank! as old Mattu pulled the rope within.'",
    },
  ],
  "17:59": [
    {
      time: "nearly six o'clock",
      book: "Dracula",
      author: "Bram Stoker",
      prefix: "When he arrived it was ",
      suffix:
        ", and the sun was setting full and warm, and the red light streamed in through the window and gave more colour to the pale cheeks.",
    },
  ],
  "18:00": [
    {
      time: "6.00 p.m.",
      book: "The Curious Incident of the Dog in the Night-time",
      author: "Mark Haddon",
      prefix: "",
      suffix: " Have te",
    },
    {
      time: "six o'clock",
      book: "Le Club des Hachichins",
      author: "Théophile Gautier",
      prefix: "Although it was only ",
      suffix:
        ", the night was already dark. The fog, made thicker by its proximity to the Seine, blurred every detail with its ragged veils, punctured at various distances by the reddish glow of lanterns and bars of light escaping from illuminated windows.",
    },
    {
      time: "six o'clock",
      book: "The Go-Between",
      author: "L.P. Hartley",
      prefix:
        "Did you go down to the farm while I was away?' 'No,' I said 'but I saw Ted.' 'Did he have a message for me ?' she asked. 'He said today was no good as he was going to Norwich. But Friday at ",
      suffix:
        ", same as usual.' 'Are you sure he said six o'clock?' she asked, puzzled. 'Quite sure.'",
    },
    {
      time: "six o'clock",
      book: "Richard III",
      author: "William Shakespeare",
      prefix: "King Richard: What is o'clock? Catesby: It is ",
      suffix:
        ", full supper time. King Richard: I will not sup tonight. Give me some ink and paper.",
    },
    {
      time: "six o'clock",
      book: "Madame Bovary",
      author: "Gustave Flaubert",
      prefix: "Leon waited all day for ",
      suffix:
        " to arrive; when he got to the inn, he found no one there but Monsieur Binet, already at the table.",
    },
    {
      time: "Six o'clock",
      book: "The skin of our teeth",
      author: "Thornton Wilder",
      prefix: "Oh oh oh. ",
      suffix: " and the master not home yet.",
    },
    {
      time: "six o'clock",
      book: "The Prime of Miss Jean Brodie",
      author: "Muriel Spark",
      prefix: "The newspaper snaked through the door and there was suddenly a ",
      suffix: " feeling in the house",
    },
    {
      time: "Six o'clock",
      book: "Preludes",
      author: "T S Eliot",
      prefix:
        "The winter evening settles down With smell of steaks in passageways. ",
      suffix: ".",
    },
    {
      time: "six",
      book: "The Interpretation Of Murder",
      author: "Jed Rubenfeld",
      prefix: "When the bells of Calvary Church struck ",
      suffix:
        ", she saw Mr and Mrs Biggs hurrying down the front stoop, rushing off to the shops before they closed.",
    },
  ],
  "18:03": [
    {
      time: "three minutes past six",
      book: "The Day of the Triffids",
      author: "John Wyndham",
      prefix:
        "Above it all rose the Houses of Parliament, with the hands of the clock stopped at ",
      suffix:
        ". It was difficult to believe that all that meant nothing any more, that now it was just a pretentious confection that could decay in peace.",
    },
  ],
  "18:04": [
    {
      time: "Four minutes after six",
      book: "The Loves of Alonzo Fitz Clarence and Rosannah Ethelton",
      author: "Mark Twain",
      prefix:
        '"We will make record of it, my Rosannah; every year, as this dear hour chimes from the clock, we will celebrate it with thanksgivings, all the years of our life." "We will, we will, Alonzo!" "',
      suffix: ", in the evening, my Rosannah...”",
    },
  ],
  "18:05": [
    {
      time: "five past six",
      book: "A Glass of Blessings",
      author: "Barbara Pym",
      prefix: "At about ",
      suffix: " Piers came in carrying an evening paper and a few books.",
    },
  ],
  "18:08": [
    {
      time: "6:08 p.m.",
      book: "The Night of the Generals",
      author: "Hans Hellmut Kirst",
      prefix: "",
      suffix:
        ' The code-word "Valkyrie" reached von Seydlitz Gabler\'s headquarter',
    },
  ],
  "18:10": [
    {
      time: "six ten",
      book: "The Quiet American",
      author: "Graham Greene",
      prefix: "'Let me see now. You had a drink at the Continental at ",
      suffix:
        ".' 'Yes.' 'And at six forty-five you were talking to another journalist at the door of the Majestic?' 'Yes, Wilkins. I told you all this, Vigot, before. That night.'",
    },
  ],
  "18:15": [
    {
      time: "Quarter past six",
      book: "A Handful of Dust",
      author: "Evelyn Waugh",
      prefix: "'",
      suffix: ",' said Tony. 'He's bound to have told her by now.'",
    },
    {
      time: "quarter past six",
      book: "The Photograph",
      author: "Penelope Lively",
      prefix: "At a ",
      suffix: " he was through with them.",
    },
    {
      time: "6.15 pm.",
      book: "Girl Missing",
      author: "Sophie McKenzie",
      prefix: "I checked the time on the corner of my screen. ",
      suffix: " I was never going to finisah my essay in forty-five minutes",
    },
  ],
  "18:20": [
    {
      time: "twenty past six",
      book: "Interpreter of Maladies",
      author: "Jhumpa Lahiri",
      prefix: "By the time Elliot's mother arrived at ",
      suffix:
        ", Mrs Sen always made sure all evidence of her chopping was disposed of.",
    },
  ],
  "18:21": [
    {
      time: "6.21pm",
      book: "Miss Pettigrew Lives for a Day",
      author: "Winifred Watson",
      prefix: "5.20pm - ",
      suffix:
        ": Miss Pettigrew found herself wafted into the passage. She was past remonstrance now, past bewilderment, surprise, expostulation. Her eyes shone. Her face glowed. Her spirits soared. Everything was happening too quickly. She couldn't keep up with things, but, by golly, she could enjoy them.",
    },
  ],
  "18:22": [
    {
      time: "Twenty-two minutes past six",
      book: "The Murder at the Vicarage",
      author: "Agatha Christie",
      prefix:
        "Clock overturned when he fell forward. That'll give us the time of the crime. ",
      suffix: ".",
    },
  ],
  "18:25": [
    {
      time: "twenty-five past six",
      book: "A Kind of Loving",
      author: "Stan Barstow",
      prefix: "At ",
      suffix:
        " I go into the bathroom and have a wash, then while the Old Lady's busy in the kitchen helping Chris with the washing up I get my coat and nip out down the stairs.",
    },
    {
      time: "6.25",
      book: "Dracula",
      author: "Bram Stoker",
      prefix:
        "I have this moment, while writing, had a wire from Jonathan saying that he leaves by the ",
      suffix:
        " tonight from Launceston and will be here at 10.18, so that I shall have no fear tonight.",
    },
  ],
  "18:26": [
    {
      time: "around half past six",
      book: "Long Day's Journey Into Night",
      author: "Eugene O'Neill",
      prefix: "It is ",
      suffix:
        " in the evening. Dusk is gathering in the living room, an early dusk due to the fog which has rolled in from the Sound and is like a white curtain drawn down outside the windows.",
    },
  ],
  "18:30": [
    {
      time: "six-thirty",
      book: "The Rum Diary",
      author: "Hunter S. Thompson",
      prefix: "At ",
      suffix:
        " I left the bar and walked outside. It was getting dark and the big Avenida looked cool and graceful. On the other side were homes that once looked out on the beach. Now they looked out on hotels and most of them had retreated behind tall hedges and walls that cut them off from the street.",
    },
    {
      time: "6.30 p.m.",
      book: "The Curious Incident of the Dog in the Night-time",
      author: "Mark Haddon",
      prefix: "",
      suffix: " Watch television or a vide",
    },
    {
      time: "half-past six",
      book: "The Go-Between",
      author: "L.P. Hartley",
      prefix:
        "As I was turning away, grieved to be parting from him, a thought started up in me and I turned back. 'Shall I take one more message for you?' 'That's good of you' he said, 'but do you want to?' 'Yes, just this once.' It could do no harm, I thought; and I should be far away when the message takes effect, and I wanted to say something to show we were friends. 'Well,' he said, once more across the gap, 'say tomorrow's no good, I'm going to Norwich, but Friday at ",
      suffix: ", same as usual.'",
    },
    {
      time: "half past six",
      book: "Pride and Prejudice",
      author: "Jane Austen",
      prefix: "At five o'clock the two ladies retired to dress, and at ",
      suffix: " Elizabeth was summoned to dinner.",
    },
    {
      time: "six thirty",
      book: "Jealousy",
      author: "Alain Robbe-Grillet",
      prefix: "It is ",
      suffix:
        ". Now the dark night and the deafening racket of the crickets again engulf the garden and the veranda, all around the house",
    },
    {
      time: "half-past six",
      book: "Cotillion",
      author: "Georgette Heyer",
      prefix:
        "To a casual visitor it might have seemed that Mr Penicuik, who owned the house, had fallen upon evil days; but two of the three gentlemen assembled in the Saloon at ",
      suffix:
        " on a wintry evening of late February were in no danger of falling into this error.",
    },
  ],
  "18:31": [
    {
      time: "a little after half past six",
      book: "The Adventure of The Blue Carbuncle",
      author: "Arthur Conan Doyle",
      prefix: "I had been delayed at a case and it was ",
      suffix: " when I found myself at Baker Street once more",
    },
  ],
  "18:32": [
    {
      time: "around half past six",
      book: "Long Day's Journey Into Night",
      author: "Eugene O'Neill",
      prefix: "It is ",
      suffix:
        " in the evening. Dusk is gathering in the living room, an early dusk due to the fog which has rolled in from the Sound and is like a white curtain drawn down outside the windows.",
    },
  ],
  "18:33": [
    {
      time: "6.33pm",
      book: "Atomised",
      author: "Michel Houellebecq",
      prefix:
        "Every evening, Michel took the train home, changed at Esbly and usually arrived in Crécy on the ",
      suffix: " train where Annabelle would be waiting at the station.",
    },
  ],
  "18:34": [
    {
      time: "around half past six",
      book: "Long Day's Journey Into Night",
      author: "Eugene O'Neill",
      prefix: "It is ",
      suffix:
        " in the evening. Dusk is gathering in the living room, an early dusk due to the fog which has rolled in from the Sound and is like a white curtain drawn down outside the windows.",
    },
  ],
  "18:35": [
    {
      time: "6.35 pm",
      book: "The Curious Incident Of The Dog In The Night-Time",
      author: "Mark Haddon",
      prefix: "And then it was ",
      suffix:
        " and I heard Father come home in his van and I moved the bed up against the door so he couldn't get in and he came into the house and he and Mother shouted at each other.",
    },
  ],
  "18:36": [
    {
      time: "6:36",
      book: "The Voices of Time",
      author: "JG Ballard",
      prefix:
        "Kaldren pursues me like luminescent shadow. He has chalked up on the gateway '96,688,365,498,702'. Should confuse the mail man. Woke 9:05. To sleep ",
      suffix: ".",
    },
  ],
  "18:40": [
    {
      time: "twenty to seven",
      book: "The Family Reunion",
      author: "TS Eliot",
      prefix:
        "Amy: What's that? I thought I saw someone pass the window. What time is it? Charles: Nearly ",
      suffix: ".",
    },
    {
      time: "twenty minutes to seven",
      book: "Diary of a Nobody",
      author: "George and Weedon Grossmith",
      prefix:
        "Having to change 'buses, I allowed plenty of time — in fact, too much; for we arrived at ",
      suffix:
        ", and Franching, so the servant said, had only just gone up to dress.",
    },
  ],
  "18:41": [
    {
      time: "six forty-one",
      book: "The New York Trilogy",
      author: "Paul Auster",
      prefix:
        "He made it to Grand Central well in advance. Stillman's train was not due to arrive until ",
      suffix:
        ", but Quinn wanted time to study the geography of the place, to make sure that Stillman would not be able to slip away from him.",
    },
  ],
  "18:45": [
    {
      time: "six forty-five",
      book: "The Quiet American",
      author: "Graham Greene",
      prefix:
        "'Let me see now. You had a drink at the Continental at six ten.' 'Yes.' 'And at ",
      suffix:
        " you were talking to another journalist at the door of the Majestic?' 'Yes, Wilkins. I told you all this, Vigot, before. That night.'",
    },
    {
      time: "Six forty-five",
      book: "The Man Who Loved Children",
      author: "Christina Stead",
      prefix: '"',
      suffix:
        '," called Louie. "Did you hear, Ming," he asked, "did you hear?" "Yes, Taddy, I heard." "What is it?\' asked Tommy. "The new baby, listen, the new baby."',
    },
    {
      time: "quarter to seven",
      book: "The High Window",
      author: "Raymond Chandler",
      prefix: "It was a ",
      suffix:
        " when I let myself into the office and clicked the light on and picked a piece of paper off the floor. It was a notice from the Green Feather Messenger Service ...",
    },
  ],
  "18:49": [
    {
      time: "6:49 p.m.",
      book: "The Night of the Generals",
      author: "Hans Hellmut Kirst",
      prefix: "",
      suffix:
        " Lieutenant-General Tanz escorted by a motorized unit, drove to Corps headquarter",
    },
  ],
  "18:50": [
    {
      time: "ten minutes to seven",
      book: "The Four Million",
      author: "O. Henry",
      prefix: "At ",
      suffix:
        " Dulcie was ready. She looked at herself in the wrinkly mirror. The reflection was satisfactory. The dark blue dress, fitting without a wrinkle, the hat with its jaunty black feather, the but-slightly-soiled gloves--all representing self- denial, even of food itself--were vastly becoming. Dulcie forgot everything else for a moment except that she was beautiful, and that life was about to lift a corner of its mysterious veil for her to observe its wonders. No gentleman had ever asked her out before. Now she was going for a brief moment into the glitter and exalted show.",
    },
    {
      time: "ten minutes before seven",
      book: "Boy's Life",
      author: "Robert R. McCammon",
      prefix:
        "It was time to go see the Lady. When we arrived at her house at ",
      suffix: " o'clock, Damaronde answered the door.",
    },
    {
      time: "ten minutes before seven",
      book: "Boy's Life",
      author: "Robert R. McCammon",
      prefix:
        "It was time to go see the Lady. When we arrived at her house at ",
      suffix: " o'clock, Damaronde answered the door.",
    },
  ],
  "18:51": [
    {
      time: "6:51",
      book: "Salem's Lot",
      author: "Stephen King",
      prefix:
        "The square of light in the kitchen doorway had faded to thin purple; his watch said ",
      suffix: ".",
    },
  ],
  "18:53": [
    {
      time: "near on seven o'clock",
      book: "All Aunt Hagar's Children",
      author: "Edward P Jones",
      prefix: "It was ",
      suffix:
        " when I got to Mr. and Mrs. Fleming's house on 6th Street, where I was renting a room. It was late September, and though there was some sun left, I didn't want to visit a dead man's place with night coming on.",
    },
  ],
  "18:55": [
    {
      time: "five to seven",
      book: "The Quiet American",
      author: "Graham Greene",
      prefix:
        "... You had no reason to think the times important. Indeed how suspicious it would be if you had been completely accurate. ''Haven't I been?'' Not quite. It was ",
      suffix: " that you talked to Wilkins. ''Another ten minutes.\"",
    },
    {
      time: "6:55",
      book: "Middlesex",
      author: "Jeffrey Eugenides",
      prefix:
        "The play was set to begin at seven o'clock and finish before sunset. It was ",
      suffix:
        ". Beyond the flats we could hear the hockey field filling up. the low rumble got steadily louder - voices, footsteps, the creaking of bleachers, the slamming of car doors in the parking lot.",
    },
  ],
  "18:56": [
    {
      time: "6.56",
      book: "Dreams of leaving",
      author: "Rupert Thomson",
      prefix: "Then it was ",
      suffix:
        ". A black Rover - a Rover 90, registration PYX 520 - turned into the street that ran down the left-hand side of The Bunker. It parked. The door on the driver's side opened. A man got out.",
    },
  ],
  "18:57": [
    {
      time: "a few minutes before seven",
      book: "Bridging",
      author: "Max Apple",
      prefix:
        '"I feel a little awkward," Kay Randall said on the phone, "asking a man to do these errands ... but that\'s my problem, not yours. Just bring the supplies and try to be at the church meeting room ',
      suffix: '."',
    },
    {
      time: "three minutes to the hour; which was seven",
      book: "Between the Acts",
      author: "Virginia Woolf",
      prefix:
        "Folded in this triple melody, the audience sat gazing; and beheld gently and approvingly without interrogation, for it seemed inevitable, a box tree in a green tub take the place of the ladies’ dressing-room; while on what seemed to be a wall, was hung a great clock face; the hands pointing to ",
      suffix: ".'",
    },
  ],
  "18:58": [
    {
      time: "two minutes to seven",
      book: "Roads of Destiny",
      author: "O. Henry",
      prefix: '"Walk fast," says Perry, "it\'s ',
      suffix:
        ', and I got to be home by—\' "Oh, shut up," says I. "I had an appointment as chief performer at an inquest at seven, and I\'m not kicking about not keeping it."',
    },
  ],
  "18:59": [
    {
      time: "About seven o’clock",
      book: "Herbert West - Reanimator",
      author: "H. P. Lovecraft",
      prefix: "",
      suffix:
        " in the evening she had died, and her frantic husband had made a frightful scene in his efforts to kill West, whom he wildly blamed for not saving her life. Friends had held him when he drew a stiletto, but West departed amidst his inhuman shrieks, curses, and oaths of vengeance",
    },
  ],
  "19:00": [
    {
      time: "seven o'clock",
      book: "In Search of Lost Time: Swann's Way",
      author: "Marcel Proust",
      prefix:
        "… in a word, seen always at the same evening hour, isolated from all its possible surroundings, detached and solitary against its shadowy background, the bare minimum of scenery necessary .. to the drama of my undressing, as though all Combray had consisted of but two floors joined by a slender staircase, and as though there had been no time there but ",
      suffix: " at night.",
    },
    {
      time: "seven",
      book: "Something Wicked This Way Comes",
      author: "Ray Bradbury",
      prefix: "The town clock struck ",
      suffix:
        ". The echoes of the great chime wandered in the unlit halls of the library. An autumn leaf, very crisp, fell somewhere in the dark. But it was only the page of a book, turning.",
    },
    {
      time: "7.00 p.m.",
      book: "The Curious Incident of the Dog in the Night-time",
      author: "Mark Haddon",
      prefix: "",
      suffix: " Do maths practic",
    },
    {
      time: "seven o'clock",
      book: "The Great Gatsby",
      author: "F. Scott Fitzgerald",
      prefix: "By ",
      suffix:
        " the orchestra has arrived--no thin five-piece affair but a whole pitful of oboes and trombones and saxophones and viols and cornets and piccolos and low and high drums.",
    },
    {
      time: "seven",
      book: "New Moon",
      author: "Stephenie Meyer",
      prefix: "Edward had been allowed to see me only from ",
      suffix:
        " till nine-thirty pm, always inside the confines of my home and under the supervision of my dad's unfailingly crabby glare.",
    },
    {
      time: "seven o'clock",
      book: "Silas Marner",
      author: "George Eliot",
      prefix: "It was ",
      suffix:
        " and by this time she was not very far from Raveloe, but she was not familiar enough with those monotonous lanes to know how near she was to her journey's end. She needed comfort, and she knew but one comforter - the familiar demon in her bosom; but she hesitated a moment, after drawing out the black remnant, before she raised it to her lips.",
    },
    {
      time: "seven o'clock",
      book: "The Great Gatsby",
      author: "F. Scott Fitzgerald",
      prefix: "It was ",
      suffix:
        " when we got into the coupé with him and started for Long Island. [...] So we drove on toward death through the cooling twilight.",
    },
  ],
  "19:01": [
    {
      time: "around seven",
      book: "The Talented Mr Ripley",
      author: "Patricia Highsmith",
      prefix: "He waited until nearly eight, because ",
      suffix:
        " there were always more people coming in and out of the house than at other times.",
    },
  ],
  "19:02": [
    {
      time: "about seven o'clock",
      book: "The Tay Bridge Disaster",
      author: "William McGonagall",
      prefix: "Twas ",
      suffix:
        " at night, And the wind it blew with all its might, And the rain came pouring down, And the dark clouds seem'd to frown,",
    },
  ],
  "19:08": [
    {
      time: "eight minutes past seven",
      book: "The Girl from East Berlin",
      author: "James Furner",
      prefix: "It was ",
      suffix:
        " and still no girl. I waited impatiently. I watched another crowd surge through the barriers and move quickly down the steps. My eyes were alert for the faintest recognition.",
    },
  ],
  "19:10": [
    {
      time: "in five minutes it would be a quarter past seven",
      book: "Metamorphosis",
      author: "Franz Kafka",
      prefix:
        "He had already got to the point where, by rocking more strongly, he maintained his equilibrium with difficulty, and very soon he would finally have to make a final decision, for ",
      suffix:
        ". Then there was a ring at the door of the apartment. “That’s someone from the office,” he told himself, and he almost froze, while his small limbs only danced around all the faster. For one moment everything remained still. “They aren’t opening,” Gregor said to himself, caught up in some absurd hope.",
    },
    {
      time: "seven-ten",
      book: "The Elderly Lady",
      author: "Jorge Luis Borges",
      prefix:
        "The party was to begin at seven. The invitations gave the hour as six-thirty because the family knew everyone would come a little late, so as not to be the first to arrive. At ",
      suffix:
        " not a soul had come; somewhat acrimoniously, the family discussed the advantages and disadvantages of tardiness",
    },
  ],
  "19:11": [
    {
      time: "19:11",
      book: "The Whole Story and Other Stories",
      author: "Ali Smith",
      prefix:
        "Good, you said. Run, or you won't get a seat. See you soon. Your voice was reassuring. ",
      suffix:
        ":00, the clock said. I put the phone back on its hook and I ran. The seat I got, almost the last one in the carriage, was opposite a girl who started coughing as soon as there weren't any other free seats I could move to. She looked pale and the cough rattled deep in her chest as she punched numbers into her mobile. Hi, she said (cough). I'm on the train. No, I've got a cold. A cold (cough). Yeah, really bad. Yeah, awful actually. Hello? (cough) Hello?",
    },
  ],
  "19:12": [
    {
      time: "7:12",
      book: "A Fraction of the Whole",
      author: "Steve Toltz",
      prefix:
        "He taught me that if I had to meet someone for an appointment, I must refuse to follow the 'stupid human habit' of arbitrarily choosing a time based on fifteen-minute intervals. \"Never meet people at 7:45 or 6:30, Jasper, but pick times like ",
      suffix: ' and 8:03!"',
    },
  ],
  "19:14": [
    {
      time: "7:14",
      book: "The Life and Opinions of Maf the Dog, and of his friend Marilyn Monroe",
      author: "Andrew O'Hagan",
      prefix:
        "If he'd lived in New York and worked in an office, he might have thrived as the typical, over-martini'd, cheating husband, leaving every night on the ",
      suffix:
        " to White Plains, a smudge of lipstick high on his neck, and a tide of lies to see him through to the next day.",
    },
  ],
  "19:15": [
    {
      time: "7:15",
      book: "The Voices of Time",
      author: "JG Ballard",
      prefix: "Cell count down to 400,000. Woke 8:10. To sleep ",
      suffix:
        ". (Appear to have lost my watch without realising it, had to drive into town to buy another.)",
    },
    {
      time: "seven fifteen",
      book: "The Line of Beauty",
      author: "Alan Hollinghurst",
      prefix:
        "Nick had a large wild plan of his own for the night, but for now he let Leo take charge: they were going to go back to Notting Hill and catch the ",
      suffix: " screening of Scarface at the Gate.",
    },
    {
      time: "seven-fifteen",
      book: "The elderly lady",
      author: "Jorge Luis Borges",
      prefix:
        "The party was to begin at seven. The invitations gave the hour as six-thirty because the famly knew everyone would come a little late, so as not to be the first to arrive. .. By ",
      suffix: " not another soul could squeeze into the house.",
    },
  ],
  "19:16": [
    {
      time: "Sixteen past seven PM",
      book: "The Last Precinct",
      author: "Patricia Cornwell",
      prefix: "“",
      suffix:
        "? That's when he came into the store or when he left after the fact?”",
    },
  ],
  "19:17": [
    {
      time: "7.17 p.m.",
      book: "The Dogs of Riga",
      author: "Henning Mankell",
      prefix: "Colonel Putnis knocked on his door at ",
      suffix:
        " The car was waiting in front of the hotel, and they drove through the dark streets to police headquarters. It had grown much colder during the evening, and the city was almost deserted.",
    },
  ],
  "19:19": [
    {
      time: "seven-nineteen",
      book: "The Undomestic Goddess",
      author: "Sophie Kinsella",
      prefix:
        "And it was me who spent about three hours this afternoon arguing one single contract. The term was best endeavors. The other side wanted to use reasonable efforts. In the end we won the point- but I can't feel my usual triumph. All I know is, it's ",
      suffix:
        ", and in eleven minutes I'm supposed to be halfway across town, sitting down to dinner at Maxim's with my mother and brother Daniel. I'll have to cancel. My own birthday dinner.",
    },
  ],
  "19:20": [
    {
      time: "seven-twenty",
      book: "Hard Boiled Wonderland and the End of the World",
      author: "Haruki Murakami",
      prefix: "The clock read ",
      suffix:
        ", but I felt no hunger. You'd think I might have wanted to eat something after the day I'd had, but I cringed at the very thought of food. I was short of sleep, my gut was slashed, and my apartment was gutted. There was no room for appetite.",
    },
    {
      time: "7:20",
      book: "Parkinson's Law or the Pursuit of Progress",
      author: "C Northcote Parkinson",
      prefix:
        "The pause, we finally concluded, was to allow the other important people to catch up, those who had arrived at 7:10 waiting for those who had arrived at ",
      suffix: ".",
    },
  ],
  "19:21": [
    {
      time: "7:21",
      book: "1Q84",
      author: "Haruki Murakami",
      prefix:
        "Gripping her gym bag in her right hand, Aomame, like Buzzcut, was waiting for something to happen. The clock display changed to ",
      suffix: ", then 7:22, then 7:23.",
    },
  ],
  "19:22": [
    {
      time: "7:22",
      book: "1Q84",
      author: "Haruki Murakami",
      prefix:
        "Gripping her gym bag in her right hand, Aomame, like Buzzcut, was waiting for something to happen. The clock display changed to 7:21, then ",
      suffix: ", then 7:23.",
    },
  ],
  "19:23": [
    {
      time: "7:23",
      book: "1Q84",
      author: "Haruki Murakami",
      prefix:
        "Gripping her gym bag in her right hand, Aomame, like Buzzcut, was waiting for something to happen. The clock display changed to 7:21, then 7:22, then ",
      suffix: ".",
    },
  ],
  "19:24": [
    {
      time: "almost twenty-five after seven",
      book: "The Evening's at Seven",
      author: "James Thurber",
      prefix:
        "He picked up his hat and coat and Clarice said hello to him and he said hello and looked at the clock and it was ",
      suffix: ".",
    },
  ],
  "19:25": [
    {
      time: "twenty-five after seven",
      book: "The Evening's at Seven",
      author: "James Thurber",
      prefix:
        "He picked up his hat and coat and Clarice said hello to him and he said hello and looked at the clock and it was almost ",
      suffix: ".",
    },
  ],
  "19:30": [
    {
      time: "half-past seven",
      book: "Crime and Punishment",
      author: "Fyodor Dostoyevsky",
      prefix:
        "But now he was close - here was the house, here were the gates. Somewhere a clock beat a single chime. 'What, is it really ",
      suffix: "? That's impossible, it must be fast!’",
    },
    {
      time: "7:30",
      book: "The Terrors of Ice and Darkness",
      author: "Christoph Ransmayr",
      prefix:
        "On July 25th, 8:30 a.m. the bitch Novaya dies whelping. At 10 o'clock she is lowered into her cool grave, at ",
      suffix:
        " that same evening we see our first floes and greet them wishing they were the last.",
    },
    {
      time: "half-past seven",
      book: "Danny, the Champion of the World",
      author: "Roald Dahl",
      prefix: "The clock showed ",
      suffix:
        ". This was the twilight time. He would be there now. I pictured him in his old navy-blue sweater and peaked cap, walking soft-footed up the track towards the wood. He told me he wore the sweater because navy-blue barely showed up in the dark, black was even better, he said. The peaked cap was important too, he explained, because the peak casts a shadow over one's face.",
    },
    {
      time: "7.30",
      book: "A Simple Story",
      author: "Leonardo Sciascia",
      prefix: "The telephone call came at ",
      suffix:
        " on the evening of March 18th, a Saturday, the eve of the noisy, colourful festival that the town held in honour of Saint Joseph the carpenter -",
    },
  ],
  "19:35": [
    {
      time: "7.35",
      book: "The Case of the Gilded Fly",
      author: "Edmund Crispin",
      prefix: "",
      suffix: "-40. Yseut arrives at 'M. and S.', puts through phone call",
    },
  ],
  "19:40": [
    {
      time: "7.40",
      book: "The Babysitter",
      author: "Robert Coover",
      prefix: "She arrives at ",
      suffix:
        ", ten minutes late, but the children, Jimmy and Bitsy, are still eating supper and their parents are not ready to go yet. From other rooms come the sound of a baby screaming, water running, a television musical (no words: probably a dance number - patterns of gliding figures come to mind).",
    },
  ],
  "19:42": [
    {
      time: "Seven forty-two",
      book: "The Undomestic Goddess",
      author: "Sophie Kinsella",
      prefix: "I glance at my watch as we speed along the Strand. ",
      suffix:
        ". I'm starting to feel quite excited. The street outside is still bright and warm and tourists are walking along in T-shirts and shorts, pointing at the High Court. It must have been a gorgeous summer's day. Inside the air-conditioned Carter Spink building you have no idea what the weather in the real world is doing.",
    },
  ],
  "19:45": [
    {
      time: "7:45",
      book: "A Fraction of the Whole",
      author: "Steve Toltz",
      prefix:
        "He taught me that if I had to meet someone for an appointment, I must refuse to follow the 'stupid human habit' of arbitrarily choosing a time based on fifteen-minute intervals. \"Never meet people at ",
      suffix: ' or 6:30, Jasper, but pick times like 7:12 and 8:03!"',
    },
    {
      time: "19.45",
      book: "Other People's Money",
      author: "Justin Cartwright",
      prefix: "He tells his old friend the train times and they settle on the ",
      suffix:
        " arriving at 23.27. 'I'll book us into the ultra-luxurious Francis Drake Lodge. Running water in several rooms. Have you got a mobile?\"",
    },
  ],
  "19:49": [
    {
      time: "eleven minutes to eight",
      book: "In the Mayor's Parlour",
      author: "J. S. Fletcher",
      prefix:
        'There\'s a big, old-fashioned clock in the surgery. Just as Dr. Wellesley went out I heard the Moot Hall clock chime half-past seven, and then the chimes of St. Hathelswide\'s Church. I noticed that our clock was a couple of minutes slow, and I put it right." When did you next see Dr. Wellesley?" "At just ',
      suffix:
        '." "Where?" "In the surgery." "He came back there?" "Yes." "How do you fix that precise time--eleven minutes to eight?" "Because he\'d arranged to see a patient in Meadow Gate at ten minutes to eight. I glanced at the clock as he came in, saw what time it was, and reminded him of the appointment."',
    },
  ],
  "19:50": [
    {
      time: "ten to eight",
      book: "The Talented Mr Ripley",
      author: "Patricia Highsmith",
      prefix: "At ",
      suffix:
        ", he strolled downstairs, to make sure that Signora Buffi was not pottering around in the hall and that her door was not open, and to make sure there really was no one in Freddie's car",
    },
  ],
  "19:52": [
    {
      time: "nearly eight",
      book: "The Talented Mr Ripley",
      author: "Patricia Highsmith",
      prefix: "He waited until ",
      suffix:
        ", because around seven there were always more people coming in and out of the house than at other times. At ten to eight, he strolled downstairs, to make sure that Signora Buffi was not pottering around in the hall and that her door was not open, and to make sure there really was no one in Freddie's car, though he had gone down in the middle of the afternoon to look at the car and see if it was Freddie's.",
    },
  ],
  "19:53": [
    {
      time: "7.53 p.m.",
      book: "A Place of Execution",
      author: "Val McDermid",
      prefix: "Wednesday, 11 th December 1963. ",
      suffix:
        " \"Help me. You've got to help me.\" The woman's voice quavered on the edge of tears. The duty constable who had picked up the phone heard a hiccuping gulp, as if the caller was struggling to speak.",
    },
  ],
  "19:54": [
    {
      time: "six minutes to eight",
      book: "A Man Lay Dead",
      author: "Ngaio Marsh",
      prefix: "The body was found at ",
      suffix:
        ". Doctor Young arrived some thirty minutes later. Just let me get that clear - I've a filthy memory.",
    },
  ],
  "19:55": [
    {
      time: "five to eight",
      book: "Cold Comfort Farm",
      author: "Stella Gibbons",
      prefix:
        "Flora drew her coat round her, and looked up into the darkening vault of the sky. Then she glanced at her watch. It was ",
      suffix: ".",
    },
  ],
  "19:56": [
    {
      time: "four minutes to eight",
      book: "Funes the Memorious-Labyrinths",
      author: "Jorge Luis Borges",
      prefix:
        "I remember the cigarette in his hard face, against the now limitless storm cloud. Bernardo cried to him unexpectedly: 'What time is it, Ireno?' Without consulting the sky, without stopping, he replied: 'It's ",
      suffix: ", young Bernardo Juan Franciso.' His voice was shrill, mocking.",
    },
  ],
  "19:57": [
    {
      time: "three minutes till eight",
      book: "Carter Beats The Devil",
      author: "Glen David Gold",
      prefix: "At ",
      suffix:
        ', Laszlo and His Yankee Hussars set up onstage. While the band played their Sousa medley, Carter thoroughly checked his kit, stuffing his pockets with scarves, examining the seals on decks of cards. He glanced toward his levitation device. "Good luck, Carter." The voice was quiet.',
    },
  ],
  "19:58": [
    {
      time: "7.58pm.",
      book: "The Lost Symbol",
      author: "Dan Brown",
      prefix: "Robert Langdon stole an anxious glance at his wristwatch: ",
      suffix: " The smiling face of Mickey Mouse did little to cheer him up.",
    },
  ],
  "19:59": [
    {
      time: "just before eight o' clock",
      book: "The Maker of Heavenly Trousers",
      author: "Daniel Vare",
      prefix: "Kuniang made her appearance in my study ",
      suffix: ', arrayed in what had once ben a "party frock".',
    },
    {
      time: "A minute to eight.",
      book: "The Thirteenth Tale",
      author: "Diane Setterfield",
      prefix: "Quickly, quickly. ",
      suffix:
        " My hot water bottle was ready, and I filled a glass with water from the tap. Time was of the essence.",
    },
  ],
  "20:00": [
    {
      time: "eight o'clock",
      book: "The Idiot Boy",
      author: "William Wordsworth",
      prefix: "'TIS ",
      suffix:
        ",--a clear March night, The moon is up,--the sky is blue, The owlet, in the moonlight air, Shouts from nobody knows where; He lengthens out his lonely shout, Halloo! halloo! a long halloo!",
    },
    {
      time: "at eight",
      book: "Tell-All",
      author: "Chuck Palahniuk",
      prefix: "\"I trace the words, I'll arrive to collect you for drinks ",
      suffix: ' on Saturday."',
    },
    {
      time: "8.00 p.m.",
      book: "The Curious Incident of the Dog in the Night-time",
      author: "Mark Haddon",
      prefix: "",
      suffix: " Have a bat",
    },
    {
      time: "eight o'clock",
      book: "So Long, and Thanks for All the Fish",
      author: "Douglas Adams",
      prefix:
        "Arthur thought he could even bear to listen to the album of bagpipe music he had won. It was ",
      suffix:
        " and he decided he would make himself, force himself, to listen to the whole record before he phoned her.",
    },
    {
      time: "eight o'clock",
      book: "Satanic Verses",
      author: "Salman Rushdie",
      prefix: "At ",
      suffix:
        " that evening, a Saturday, Pamela Chamcha stood with Jumpy Joshi - who had refused to let her go unaccompanied - next to the Photo-Me machine in a corner of the main concourse of Euston station, feeling ridiculously conspiratorial.",
    },
    {
      time: "eight",
      book: "The Interpretation Of Murder",
      author: "Jed Rubenfeld",
      prefix:
        "Freud had me knock on Jung's door, to no avail. They waited until ",
      suffix: ", then set off for Brill's without him.",
    },
    {
      time: "eight o'clock",
      book: "The Great Gatsby",
      author: "F. Scott Fitzgerald",
      prefix:
        "I have been drunk just twice in my life, and the second time was that afternoon; so everything that happened has a dim, hazy cast over it. although until after ",
      suffix: " the apartment was full of cheerful sun.",
    },
    {
      time: "eight o'clock",
      book: "Life: A User's Manual",
      author: "Georges Perec",
      prefix: "It's the twenty-third of June nineteen seventy-five, and it is ",
      suffix:
        " in the evening, seated at his jigsaw puzzle, Bartlebooth has just died.",
    },
    {
      time: "eight o'clock",
      book: "The Beautiful and Damned",
      author: "F. Scott Fitzgerald",
      prefix: "She looked at her watch- it was ",
      suffix: "",
    },
    {
      time: "at eight",
      book: "Les Miserables",
      author: "Victor Hugo",
      prefix: "That day he forgot to go to dinner; he noticed the fact ",
      suffix:
        " in the evening, and as it was too late to go to the Rue St Jaques, he ate a lump of bread.",
    },
    {
      time: "eight",
      book: "Sense and Sensibility",
      author: "Jane Austen",
      prefix: "The clock struck ",
      suffix:
        ". Had it been ten, Elinor would have been convinced that at that moment she heard a carriage driving up to the house; and so strong was the persuasion that she did, in spite of the almost impossibility of their being already come, that she moved into the adjoining dressing-closet and opened a window-shutter, to be satisfied of the truth. She instantly saw that her ears had not deceived her.",
    },
  ],
  "20:01": [
    {
      time: "a little after eight o'clock",
      book: "Slaughterhouse 5",
      author: "Kurt Vonnegut",
      prefix: "It was only ",
      suffix: ", so all the shows were about silliness or murder.",
    },
  ],
  "20:02": [
    {
      time: "two minutes past eight",
      book: "Anna Karenina",
      author: "Leo Tolstoy",
      prefix:
        '"Yes, I must go to the railway station, and if he\'s not there, then go there and catch him." Anna looked at the railway timetable in the newspapers. An evening train went at ',
      suffix: '. "Yes, I shall be in time."',
    },
  ],
  "20:03": [
    {
      time: "8:03",
      book: "A Fraction of the Whole",
      author: "Steve Toltz",
      prefix:
        "He taught me that if I had to meet someone for an appointment, I must refuse to follow the 'stupid human habit' of arbitrarily choosing a time based on fifteen-minute intervals. 'Never meet people at 7:45 or 6:30, Jasper, but pick times like 7:12 and ",
      suffix: "!'",
    },
  ],
  "20:04": [
    {
      time: "8.04",
      book: "Rabbit, Run",
      author: "John Updike",
      prefix:
        "The earth seems to cast its darkness upward into the air. The farm country is somber at night. He is grateful when the lights of Lankaster merge with his dim beams. He stops at a diner who's clock says ",
      suffix: ".",
    },
  ],
  "20:05": [
    {
      time: "8.05 pm",
      book: "The Heart of a Dog",
      author: "Mikhail Bulgakov",
      prefix: "December 23rd At ",
      suffix:
        " Prof. Preobrazhensky commenced the first operation of its kind to be performed in Europe: removal under anaesthesia of a dog's testicles and their replacement by implanted human testes, with appendages and seminal ducts, taken from a 28-year-old human male",
    },
    {
      time: "five minutes past eight",
      book: "The Bostonians",
      author: "Henry James",
      prefix:
        "Ransom took out his watch, which he had adapted, on purpose, several hours before, to Boston time, and saw that the minutes had sped with increasing velocity during this interview, and that it now marked ",
      suffix: ".",
    },
  ],
  "20:06": [
    {
      time: "after eight o’clock",
      book: "The Great Gatsby",
      author: "F. Scott Fitzgerald",
      prefix:
        "I have been drunk just twice in my life, and the second time was that afternoon; so everything that happened has a dim, hazy cast over it, although until ",
      suffix: " the apartment was full of cheerful sun.",
    },
  ],
  "20:07": [
    {
      time: "8:07 pm",
      book: "The Curious Incident of the Dog in the Night-Time",
      author: "Mark Haddon",
      prefix:
        "And I could hear that there were fewer people in the little station when the train wasn't there, so I opened my eyes and I looked at my watch and it said ",
      suffix:
        " and I had been sitting on the bench for approximately 5 hours but it hadn't seemed like approximately 5 hours, except that my bottom hurt and I was hungry and thirsty.",
    },
    {
      time: "8:07",
      book: "Mistaken Identity",
      author: "Lisa Scottoline",
      prefix:
        "Bennie pulled the transcripts for that night. The first call had come in at ",
      suffix: ", with a positive ID.",
    },
  ],
  "20:10": [
    {
      time: "2010h.",
      book: "Infinite Jest",
      author: "David Foster Wallace",
      prefix: "At ",
      suffix:
        " on 1 April Y.D.A.U., the medical attache is still watching the unlabelled entertainment cartridge.",
    },
  ],
  "20:14": [
    {
      time: "fourteen minutes past eight",
      book: "Watchers",
      author: "Dean Koontz",
      prefix: "When a call came through to Dilworth’s home number at ",
      suffix:
        " o’clock, Olbier and Jones reacted with far more excitement than the situation warranted because they were desperate for action.",
    },
  ],
  "20:15": [
    {
      time: "8:15 p.m.",
      book: "Bridget Jones's Diary",
      author: "Helen Fielding",
      prefix: "",
      suffix: " Cannot locate operating instructions (for video",
    },
    {
      time: "8.15 p.m.",
      book: "The Curious Incident of the Dog in the Night-time",
      author: "Mark Haddon",
      prefix: "",
      suffix: " Get changed into pyjama",
    },
    {
      time: "quarter past eight",
      book: "The Three Sisters",
      author: "Anton Chekhov",
      prefix:
        "Natsha: I was looking to see if there wasn't a fire. It's Shrovetide, and the servant is simply beside herself; I must look out that something doesn't happen. When I came through the dining-room yesterday midnight, there was a candle burning. I couldn't get her to tell me who had lighted it. [Puts down her candle] What's the time? Andrey: [Looks at his watch] A ",
      suffix:
        ". Natasha: And Olga and Irina aren't in yet. The poor things are still at work. Olga at the teachers' council, Irina at the telegraph office...[sighs] I said to your sister this morning, \"Irina, darling, you must take care of yourself.\" But she pays no attention. Did you say it was a quarter past eight?",
    },
  ],
  "20:16": [
    {
      time: "sixteen minutes past eight",
      book: "The Rotters' Club",
      author: "Jonathan Coe",
      prefix:
        "He kissed her hand and after a while went to get two more drinks. When he got back, it was ",
      suffix: ", and Lois was humming softly along with the jukebox",
    },
  ],
  "20:17": [
    {
      time: "20.17",
      book: "Red Dwarf",
      author: "Grant Naylor",
      prefix: "",
      suffix:
        " A red warning light failed to go on in the Drive Room, beginning a chain of events which would lead, in a further twenty-three minutes, to the total annihilation of the entire crew of Red Dwarf",
    },
  ],
  "20:18": [
    {
      time: "2018 hrs",
      book: "The Russia House",
      author: "John le Carre",
      prefix: "",
      suffix:
        " Katya has arrived at the Odessa Hotel. Barley and Katya are talking in the canteen. Wicklow and one irregular observing. More",
    },
  ],
  "20:20": [
    {
      time: "8.20 p.m.",
      book: "The Curious Incident of the Dog in the Night-time",
      author: "Mark Haddon",
      prefix: "",
      suffix: " Play computer game",
    },
    {
      time: "20.20",
      book: "H.M.S. Ulysses",
      author: "Alistair MacLean",
      prefix: "At ",
      suffix:
        " all ships had completed oiling. Hove to, they had had the utmost difficulty in keeping position in that great wind; but they were infinitely safer than in the open sea",
    },
    {
      time: "twenty minutes past eight",
      book: "Diary of a Nobody",
      author: "George and Weedon Grossmith",
      prefix:
        "Knowing that the dinner was only for us six, we never dreamed it would be a full dress affair. I had no appetite. It was quite ",
      suffix: " before we sat down to dinner.",
    },
  ],
  "20:21": [
    {
      time: "8.21",
      book: "The Ipcress File",
      author: "Len Deighton",
      prefix: "At ",
      suffix:
        ', after a knock at the door, a constable said a military police vehicle had just driven into the courtyard, the driver asking for "Mr." Murray.',
    },
  ],
  "20:23": [
    {
      time: "20.23.",
      book: "The Radiant Way",
      author: "Margaret Drabble",
      prefix: "",
      suffix:
        " In a few minutes she would go down.She could have borrowed some mascara from her daughter Sally, but it was too late. She could have rung her mother in Northam, but it was too late. Seven minutes of solitude she had, and then she would descend",
    },
  ],
  "20:24": [
    {
      time: "8.24.",
      book: "Dreams of Leaving",
      author: "Rupert Thomson",
      prefix: "Peach checked his watch. ",
      suffix: " If he wasn't in a taxi in twenty minutes he'd be done for.",
    },
  ],
  "20:25": [
    {
      time: "five and twenty past eight",
      book: "The Listerdale Mystery",
      author: "Agatha Christie",
      prefix:
        "She sat down in her usual seat and smiled at her husband as he sank into his own chair opposite her. She was saved. It was only ",
      suffix: ".",
    },
  ],
  "20:27": [
    {
      time: "seven-and-twenty minutes past eight",
      book: "Aurora Floyd",
      author: "Mary Elizabeth Braddon",
      prefix: "At ",
      suffix:
        " Mrs Lofthouse was seated at Aurora's piano, in the first agonies of a prelude in six flats; a prelude which demanded such extraordinary uses of the left hand across the right, and the right over the left, and such exercise of the thumbs in all positions",
    },
  ],
  "20:29": [
    {
      time: "Twenty-nine and a half minutes past eight",
      book: "The Four Million",
      author: "O. Henry",
      prefix: '"',
      suffix:
        ', sir." And then, from habit, he glanced at the clock in the tower, and made further oration. "By George! that clock\'s half an hour fast! First time in ten years I\'ve known it to be off. This watch of mine never varies a--" But the citizen was talking to vacancy. He turned and saw his hearer, a fast receding black shadow, flying in the direction of a house with three lighted upper windows.',
    },
  ],
  "20:30": [
    {
      time: "Half-past eight",
      book: "The Listerdale Mystery",
      author: "Agatha Christie",
      prefix:
        'Alix took up a piece of needlework and began to stitch. Gerald read a few pages of his book. Then he glanced up at the clock and tossed the book away. "',
      suffix: '. Time to go down to the cellar and start work."',
    },
    {
      time: "Half-past eight",
      book: "Inniskeen Road: July Evening",
      author: "Patrick Kavanagh",
      prefix:
        "The bicycles go by in twos and threes - there's a dance on in Billy Brennan's barn tonight, and there's the half-talk code of mysteries and the wink-and-elbow language of delight. ",
      suffix:
        " and there is not a spot upon a mile of road, no shadow thrown that might turn out a man or woman,",
    },
  ],
  "20:32": [
    {
      time: "eight thirty-two",
      book: "The Four Million",
      author: "O. Henry",
      prefix:
        "At the station he captured Miss Lantry out of the gadding mob at ",
      suffix:
        '. "We mustn\'t keep mamma and the others waiting," said she. "To Wallack\'s Theatre as fast as you can drive!"',
    },
  ],
  "20:33": [
    {
      time: "20.33",
      book: "Red Dwarf",
      author: "Grant Naylor",
      prefix: "",
      suffix:
        " Navigation officer Henri DuBois knocked his black cona coffee with four sugars over his computer console keyboard. As he mopped up the coffee, he noticed three red warning blips on his monitor screen, which he wrongly assumed were the result of his spillage",
    },
  ],
  "20:35": [
    {
      time: "8:35pm.",
      book: "Bridget Jones's Diary",
      author: "Helen Fielding",
      prefix: "",
      suffix: " Found operating instructions under Hello",
    },
    {
      time: "8.35 p.m.",
      book: "Dracula",
      author: "Bram Stoker",
      prefix: "Left Munich at ",
      suffix: " on 1st May, arriving at Vienna early the next morning",
    },
    {
      time: "five and twenty to nine",
      book: "The Listerdale mystery",
      author: "Agatha Christie",
      prefix:
        "She paused reflectively. He was keenly interested now, not a doubt of it. The murderer is bound to have an interest in murder. She had gambled on that, and succeeded. She stole a glance at the clock. It was ",
      suffix: ".",
    },
  ],
  "20:36": [
    {
      time: "20.36",
      book: "Red Dwarf",
      author: "Grant Naylor",
      prefix: "",
      suffix:
        " Rimmer stood in the main wash-room on the stasis deck and combed his hair",
    },
  ],
  "20:40": [
    {
      time: "twenty minutes to nine",
      book: "Great Expectations",
      author: "Charles Dickens",
      prefix:
        "It was when I stood before her, avoiding her eyes, that I took note of the surrounding objects in detail, and saw that her watch had stopped at ",
      suffix:
        ", and that a clock in the room had stopped at twenty minutes to nine.",
    },
    {
      time: "twenty minutes to nine",
      book: "The Murder of Roger Ackroyd",
      author: "Agatha Christie",
      prefix: "The letter had been brought in at ",
      suffix: ".",
    },
  ],
  "20:42": [
    {
      time: "8.42",
      book: "Around the World in Eighty Days",
      author: "Jules Verne",
      prefix: "The hand at this moment pointed to ",
      suffix:
        ". The players took up their cards, but their eyes were constantly on the clock. One may safely say that, however secure they might feel, never had minutes seemed so long to them.",
    },
  ],
  "20:43": [
    {
      time: "8.43",
      book: "Around the world in eighty days",
      author: "Jules Verne",
      prefix: "'",
      suffix:
        ",' said Thomas Flanagan, as he cut the cards placed before him by Gauthier Ralph. There was a moment's pause, during which the spacious room was perfectly silent.",
    },
  ],
  "20:44": [
    {
      time: "8.44!",
      book: "Around the World in Eighty Days",
      author: "Jules Verne",
      prefix:
        "The clock's pendulum beat every second with mathematical regularity, and each player could count every sixtieth of a minute as it struck his ear.\"",
      suffix:
        '" said John Sullivan, in a voice that betrayed his emotion.Only one minute more and the wager would be won.',
    },
  ],
  "20:45": [
    {
      time: "8.45",
      book: "Around the World in Eighty Days",
      author: "Jules Verne",
      prefix:
        "'It's not impossible,' Phileas said quietly.'I bet you 20,000 pounds I could do it. If I leave this evening on the ",
      suffix:
        " train to Dover, I can be back here at the Reform Club by 8.45 on Saturday 21 December. I'll get my passport stamped at every place i stop to prove I've been around the world.'",
    },
    {
      time: "quarter to nine",
      book: "A Handful of Dust",
      author: "Evelyn Waugh",
      prefix: "Beaver arrived at ",
      suffix:
        " in a state of high self-approval; he had refused two invitations for dinner while dressing that evening; he had cashed a cheque for ten pounds at his club; he had booked a Divan table at Espinosa's.",
    },
  ],
  "20:46": [
    {
      time: "eight forty six",
      book: "Macedonia",
      author: "Tom Lichtenberg",
      prefix: "At the tone, the time will be ",
      suffix:
        ", exactly. One cubic mile of seawater contains about 50 pounds of gold.",
    },
  ],
  "20:49": [
    {
      time: "8.49",
      book: "Magic Bleeds",
      author: "Ilona Andrews",
      prefix: "",
      suffix:
        ". I took the phone, cleared my throat, and dialled the keep, the packs stronghold on the outskirts of Atlanta. Just keep it professional. Less pathetic that way",
    },
  ],
  "20:50": [
    {
      time: "8.50pm.",
      book: "Bridget Jones's Diary",
      author: "Helen Fielding",
      prefix: "",
      suffix:
        ' Ah Diagram "Buttons for IMC functions". But what are IMC functions',
    },
    {
      time: "ten minutes before nine",
      book: "Around the world in eighty days",
      author: "Jules Verne",
      prefix: "all the clocks in London were striking ",
      suffix: ".",
    },
    {
      time: "ten minutes to nine",
      book: "The Reluctant Widow",
      author: "Georgette Heyer",
      prefix:
        "He glanced at the bracket-clock on the mantelpiece, but as this had stopped, drew out his watch. 'It is already too late,' he said. 'It wants only ",
      suffix:
        ".' 'Good God!' she exclaimed, turning quite pale. 'What am I to do?'",
    },
    {
      time: "2050",
      book: "Infinite Jest",
      author: "David Foster Wallace",
      prefix: "He was, yes, always home from work by ",
      suffix: " on Thursdays.",
    },
    {
      time: "ten minutes to nine",
      book: "The Secret Agent",
      author: "Joseph Conrad",
      prefix:
        "What did it mean by beginning to tick so loudly all of a sudden? Its face indicated ",
      suffix: ". Mrs Verloc cared nothing for time, and the ticking went on.",
    },
  ],
  "20:53": [
    {
      time: "eight fifty-three",
      book: "The Undomestic Goddess",
      author: "Sophie Kinsella",
      prefix: "Only ",
      suffix:
        ". The partners' decision meeting starts in seven minutes. I'm not sure I can bear this.",
    },
  ],
  "20:55": [
    {
      time: "8:55pm.",
      book: "Such Great Heights",
      author: "Chris Cole",
      prefix: "And the past. The clock on the dash said ",
      suffix:
        " And the last pink shard of the sun was reaching up into the night sky, desperately trying to hold on for just one more minute.",
    },
  ],
  "20:56": [
    {
      time: "four minutes to nine",
      book: "Murder on the Orient Express",
      author: "Agatha Christie",
      prefix:
        "“No. 7 berth—a second-class. The gentleman has not yet come, and it is ",
      suffix: '."',
    },
  ],
  "20:57": [
    {
      time: "three minutes to nine",
      book: "The Four Million",
      author: "O. Henry",
      prefix:
        '"Wait," he said solemnly, "till the clock strikes. I have wealth and power and knowledge above most men, but when the clock strikes I am afraid. Stay by me until then. This woman shall be yours. You have the word of the hereditary Prince of Valleluna. On the day of your marriage I will give you $100,000 and a palace on the Hudson. But there must be no clocks in that palace--they measure our follies and limit our pleasures. Do you agree to that?" "Of course," said the young man, cheerfully, "they\'re a nuisance, anyway--always ticking and striking and getting you late for dinner." He glanced again at the clock in the tower. The hands stood at ',
      suffix: ".",
    },
  ],
  "20:58": [
    {
      time: "Two minutes to nine",
      book: "Sons and Lovers",
      author: "D H Lawrence",
      prefix: '"What time is it?" she asked, quiet, definite, hopeless. "',
      suffix: '," he replied, telling the truth with a struggle.',
    },
  ],
  "21:00": [
    {
      time: "9.00 p.m.",
      book: "The Curious Incident of the Dog in the Night-time",
      author: "Mark Haddon",
      prefix: "",
      suffix: " Watch television or a vide",
    },
    {
      time: "2100",
      book: "Infinite Jest",
      author: "David Foster Wallace",
      prefix: "At ",
      suffix: " at night it's cold out.",
    },
    {
      time: "nine o'clock",
      book: "His Last Bow An Epilogue of Sherlock Holmes",
      author: "Arthur Conan Doyle",
      prefix: "It was ",
      suffix:
        " at night upon the second of August—the most terrible August in the history of the world. One might have thought already that God's curse hung heavy over a degenerate world, for there was an awesome hush and a feeling of vague expectancy in the sultry and stagnant air",
    },
    {
      time: "nine o'clock",
      book: "The Trial",
      author: "Franz Kafka",
      prefix:
        "On the evening before K.'s thirty-first birthday - it was about ",
      suffix:
        ", when there is a lull in the streets - two gentlemen came to his apartment.",
    },
    {
      time: "nine o’clock",
      book: "A Marriage of Passion",
      author: "Katherine Mansfield",
      prefix: "On the stroke of ",
      suffix:
        " Mr. and Mrs. De Voted took their places on either side of the drawing-room fire, in attitudes of gracefully combined hospitality and unconcern, Vivian De Voted wearing a black beard and black velvet jacket buttoned over his Bohemian bosom, his lady in a flowing purple gown embroidered in divers appropriate places with pomegranates and their leaves.",
    },
    {
      time: "nine o'clock",
      book: "War and Peace",
      author: "Leo Tolstoy",
      prefix: "Shortly after ",
      suffix:
        " that evening, Weyrother drove with his plans to Kutuzov's quarters where the council of war was to be held. All the commanders of columns were summoned to the commander in chief's and with the exception of Prince Bagration, who declined to come, were all there at the appointed time.",
    },
    {
      time: "nine o'clock",
      book: "The Yiddish Policemen's Union",
      author: "Michael Chabon",
      prefix:
        "Standing in the chrome-and-tile desolation of the Polar-Shtern Kafeteria at ",
      suffix:
        " on a Friday night, in a snowstorm, he's the loneliest Jew in the Sitka District.",
    },
    {
      time: "at nine",
      book: "All the President's Men",
      author: "Bernstein & Woodward",
      prefix: "That night ",
      suffix: " the President addressed the nation.",
    },
    {
      time: "nine o'clock",
      book: "Pereira Maintains",
      author: "Antonio Tabucchi",
      prefix:
        "Then he put on a grey jacket and left the flat to make his way to Praca da Alegria. It was already ",
      suffix: ", Pereira maintains.",
    },
    {
      time: "nine o'clock",
      book: "Titus Groan",
      author: "Mervyn Peake",
      prefix: "This time, the putting on of her best hat at ",
      suffix:
        " at night with the idea of sallying forth from the castle, down the long drive and then northwards along the acacia avenue, had been enough to send her to her own doorway as though she suspected someone might be there, someone who was listening to her thoughts.",
    },
  ],
  "21:01": [
    {
      time: "about nine o'clock",
      book: "The Trial",
      author: "Franz Kafka",
      prefix: "On the evening before K.'s thirty-first birthday - it was ",
      suffix:
        ", when there is a lull in the streets - two gentlemen came to his apartment.",
    },
  ],
  "21:02": [
    {
      time: "after nine",
      book: "Edward Mills and George Benton: A Tale",
      author: "Mark Twain",
      prefix: "The good Brants did not allow the boys to play out ",
      suffix:
        " in summer evenings; they were sent to bed at that hour; Eddie honorably remained, but Georgie usually slipped out of the window toward ten, and enjoyed himself until midnight.",
    },
  ],
  "21:03": [
    {
      time: "about nine o’clock",
      book: "“The Landlady”",
      author: "Roald Dahl",
      prefix:
        "Billy Weaver had travelled down from London on the slow afternoon train, with a change at Swindon on the way, and by the time he got to Bath it was ",
      suffix:
        " in the evening and the moon was coming up out of a clear starry sky over the houses opposite the station entrance. But the air was deadly cold and the wind was like a flat blade of ice on his cheeks.",
    },
  ],
  "21:04": [
    {
      time: "9.04pm",
      book: "A Short History of Nearly Everything",
      author: "Bill Bryson",
      prefix: "At ",
      suffix:
        " trilobites swim onto the scene, followed more or less immediately by the shapely creatures of the Burgess Shale.",
    },
  ],
  "21:05": [
    {
      time: "Nine-five",
      book: "There will come soft rains",
      author: "Ray Bradbury",
      prefix: "",
      suffix:
        '. A voice spoke from the study ceiling: "Mrs. McClellan, which poem would you like this evening?". The house was silent. The voice said at last, "Since you express no preference, I shall select a poem at random.',
    },
  ],
  "21:09": [
    {
      time: "9.09",
      book: "Dreams of Leaving",
      author: "Rupert Thomson",
      prefix: "",
      suffix: ". Too late to turn around and go back. Too late, too dangerous",
    },
  ],
  "21:11": [
    {
      time: "9.11",
      book: "The Ipcress File",
      author: "Len Deighton",
      prefix:
        "Every few seconds the house changed character, at one time menacing and sinister, and again the innocent abode of law-abiding citizens about to be attacked by my private army. The luminous watch said ",
      suffix: ".",
    },
  ],
  "21:12": [
    {
      time: "21.12",
      book: "Burley Cross Postbox Theft",
      author: "Nicola Barker",
      prefix:
        "The crime was reported to us (with almost indecent alacrity, Rog) at ",
      suffix:
        ", by Susan Trott - of Black Grouse Cottage - who had been, I quote: 'out looking for hedgehogs when I was horrified to notice the postbox door had fallen off and was just lying there, on the ground'.",
    },
  ],
  "21:15": [
    {
      time: "9.15",
      book: "Ulysses",
      author: "James Joyce",
      prefix: "",
      suffix: ". Did Roberts pay you yet",
    },
    {
      time: "nine-fifteen",
      book: "Can You Keep a Secret?",
      author: "Sophie Kinsella",
      prefix:
        "What are we going to do? Should we try to walk to Clapham High Street? But it's bloody miles away. I glance at my watch and am shocked to see that it's ",
      suffix:
        ". We've spent over an hour faffing about and we haven't even had a drink. And it's all my fault. I can't even organize one simple evening without its going catastrophically wrong.",
    },
  ],
  "21:17": [
    {
      time: "21:17",
      book: "Let The Right One In",
      author: "John Ajvide Lindqvist",
      prefix: "",
      suffix:
        ", Sunday Evening, Angbyplan. A man is observed outside the hair salon. He presses his face and hands against the glass, and appears extremely intoxicated",
    },
  ],
  "21:18": [
    {
      time: "eighteen minutes after nine",
      book: "The Catbird Seat",
      author: "James Thurber",
      prefix:
        "The same thing would hold true if there were someone in her apartment. In that case he would just say that he had been passing by, recognized her charming house, and thought to drop in. It was ",
      suffix: " when Mr. Martin turned into Twelfth Street.",
    },
  ],
  "21:20": [
    {
      time: "9.20 p.m.",
      book: "The Curious Incident of the Dog in the Night-time",
      author: "Mark Haddon",
      prefix: "",
      suffix: " Have juice and a snac",
    },
  ],
  "21:22": [
    {
      time: "21.22 hrs",
      book: "Burley Cross Postbox Theft",
      author: "Nicola Barker",
      prefix: "Fifteen minutes later (",
      suffix:
        "), Miss Squire arrives in Skipton where she is booked into a local B&B. This B&B is located directly across the road from Mhairi Callaghan's Feathercuts.",
    },
  ],
  "21:23": [
    {
      time: "9.23pm",
      book: "The Secret Diary of Adrian Mole Aged 13 3/4",
      author: "Sue Townsend",
      prefix:
        "My father met me at the station, the dog jumped up to meet me, missed, and nearly fell in front of the ",
      suffix: " Birmingham express.",
    },
  ],
  "21:25": [
    {
      time: "9:25 p.m.",
      book: "Bridget Jones's Diary",
      author: "Helen Fielding",
      prefix: "",
      suffix:
        " Aargh. Suddenly main menu is on TV saying Press 6. Realize was using telly remote control by mistake. Now News has come o",
    },
  ],
  "21:28": [
    {
      time: "9:28",
      book: "Everything is Illuminated",
      author: "Jonathan Safran Foer",
      prefix: "From that moment on--",
      suffix: " in the evening, June 18, 1941--everything was different.",
    },
  ],
  "21:30": [
    {
      time: "9.30 p.m.",
      book: "The Curious Incident of the Dog in the Night-time",
      author: "Mark Haddon",
      prefix: "",
      suffix: " Go to be",
    },
    {
      time: "nine thirty",
      book: "Saturday",
      author: "Ian McEwan",
      prefix: "Forty-eight years old, profoundly asleep at ",
      suffix: " on a Friday night - this is modern professional life.",
    },
    {
      time: "9:30 p.m.",
      book: "Maus",
      author: "Art Spiegelman",
      prefix: "It's ",
      suffix:
        " already. I've gotta head uptown for my appointment with Pavel. Pavel is my shrink. He sees patients at night. He's a Czech Jew, a survivor of Terezin and Auswitz. I see him once a week.",
    },
    {
      time: "nine-thirty",
      book: "A crime in the neighborhood",
      author: "Suzanne Berne",
      prefix: "The light in Mr. Green's kitchen snapped off at ",
      suffix:
        ", followed by the light in his bedroom at his usual ten o'clock. His house was the first on the street to go dark.",
    },
  ],
  "21:31": [
    {
      time: "9:31",
      book: "A Wild Sheep Chase",
      author: "Haruki Murakami",
      prefix:
        'I took some juice out of the refrigerator and sat down at the kitchen table with it. On the table was a note from my girlfriend: "Gone out to eat. Back by 9:30." The digital clock on the table read 9:30. I watched it flip over to ',
      suffix: ", then to 9:32.",
    },
  ],
  "21:32": [
    {
      time: "9:32",
      book: "A Wild Sheep Chase",
      author: "Haruki Murakami",
      prefix:
        'I took some juice out of the refrigerator and sat down at the kitchen table with it. On the table was a note from my girlfriend: "Gone out to eat. Back by 9:30." The digital clock on the table read 9:30. I watched it flip over to 9:31, then to ',
      suffix: ".",
    },
  ],
  "21:34": [
    {
      time: "9.34 p.m.",
      book: "The Riddle of the Sands",
      author: "Erskine Childers",
      prefix: "Thanks; expect me ",
      suffix:
        " 26th'; which produced, three hours later, a reply: 'Delighted; please bring a No. 3 Rippingille stove' - a perplexing and ominous direction, which somehow chilled me in spite of its subject matter.",
    },
  ],
  "21:35": [
    {
      time: "9.35 p.m.",
      book: "A Blunt Instrument",
      author: "Georgette Heyer",
      prefix:
        "The Sergeant jotted it down on a piece of paper. 'That checks up with his own story: ",
      suffix: " Budd leaves; the North dame arrives.'",
    },
  ],
  "21:36": [
    {
      time: "9:36",
      book: "Extremely Loud and Incredibly Close",
      author: "Jonathan Safran Foer",
      prefix:
        "My backpack was already packed, and I'd already gotten the other supplies together, like the altimeter and the granola bars and the Swiss army knife I'd dug up in Central Park, so there was nothing else to do. Mom tucked me in at ",
      suffix: ".",
    },
  ],
  "21:38": [
    {
      time: "nine thirty-eight",
      book: "Pig and Pepper",
      author: "David Footman",
      prefix: "At ",
      suffix:
        " the waiter came back and offered us a second helping of cheese,salami and sardines, and Mr Yoshogi who had been converting sterling into yen looked extremely puzzled and said he had no idea that British Honduras had so large an export trade",
    },
  ],
  "21:42": [
    {
      time: "9:42 P.M.",
      book: "The Lost Symbol",
      author: "Dan Brown",
      prefix: "Langdon looked at his Mickey Mouse watch. ",
      suffix: "",
    },
  ],
  "21:45": [
    {
      time: "9:45 PM",
      book: "Riven Rock",
      author: "T. C. Boyle",
      prefix:
        "But for some unfathomable reason-birth, death, the end of the universe and all things available to man-Cody Menhoff's was closed at ",
      suffix: " on a Thursday...",
    },
  ],
  "21:47": [
    {
      time: "thirteen minutes to ten",
      book: "Elegy for a Revolutionary",
      author: "CJ Driver",
      prefix:
        "For Hunter, who was trained to note times exactly, the final emergency started at ",
      suffix: ".",
    },
  ],
  "21:50": [
    {
      time: "ten minutes to ten",
      book: "Dubliners",
      author: "James Joyce",
      prefix:
        "I passed out on to the road and saw by the lighted dial of a clock that it was ",
      suffix:
        ". In front of me was a large building which displayed the magical name.",
    },
  ],
  "21:57": [
    {
      time: "21:57",
      book: "Cloud Atlas",
      author: "David Mitchell",
      prefix: "Second to last, the inset clock blinks from ",
      suffix:
        " to 21:58. Napier's eyes sink, newborn sunshine slants through ancient oaks and on a lost river. Look, Joe, herons",
    },
    {
      time: "Three minutes to ten",
      book: "The Four Million",
      author: "O. Henry",
      prefix:
        'The waiting man pulled out a handsome watch, the lids of it set with small diamonds. "',
      suffix: '," he announced.',
    },
  ],
  "21:58": [
    {
      time: "21:58",
      book: "Cloud Atlas",
      author: "David Mitchell",
      prefix: "Second to last, the inset clock blinks from 21:57 to ",
      suffix:
        ". Napier's eyes sink, newborn sunshine slants through ancient oaks and on a lost river. Look, Joe, herons",
    },
  ],
  "21:59": [
    {
      time: "about 10",
      book: "The Life and Opinions of Tristram Shandy, Gentleman",
      author: "Laurence Sterne",
      prefix:
        "The first night, as soon as the corporal had conducted my uncle Toby upstairs, which was ",
      suffix: " - Mrs. Wadman threw herself into her arm chair",
    },
  ],
  "22:00": [
    {
      time: "ten",
      book: "The Shipping News",
      author: "E. Annie Proulx",
      prefix: "By ",
      suffix:
        ", Quoyle was drunk. The crowd was enormous, crushed together so densely that Nutbeem could not force his way down the hall or to the door and urinated on the remaining potato chips in the blue barrel, setting a popular example.",
    },
    {
      time: "Ten o'clock",
      book: "The Great Gatsby",
      author: "F. Scott Fitzgerald",
      prefix:
        "Her body asserted itself with a restless movement of the knee, and she stood up. '",
      suffix:
        ",' she remarked, apparently finding the time on the ceiling. 'Time for this good girl to go to bed.'",
    },
    {
      time: "ten",
      book: "Treasure Island",
      author: "Robert Louis Stevenson",
      prefix:
        "I could not doubt that this was the BLACK SPOT; and taking it up, I found writ",
      suffix:
        ' on the other side, in a very good, clear hand, this short message: "You have till ten tonight."',
    },
    {
      time: "ten o'clock",
      book: "Rebecca",
      author: "Daphne du Maurier",
      prefix:
        "I went back into the library, limp and exhausted. In a few minutes the telephone began ringing again. I did not do anything. I let it ring. I went and sat down at Maxim's feet. It went on ringing. I did not move. Presently it stopped, as though cut suddenly in exasperation. The clock on the mantelpiece struck ",
      suffix:
        ". Maxim put his arms round me and lifted me against him. We began to kiss one another, feverishly, desperately, like guilty lovers who have not kissed before.",
    },
    {
      time: "ten o'clock",
      book: "Little Women",
      author: "Louisa May Alcott",
      prefix: "No one wanted to go to bed when at ",
      suffix:
        ' Mrs. March put by the last finished job, and said, "Come girls." Beth went to the piano and played the father\'s favorite hymn.',
    },
    {
      time: "ten",
      book: "The Thousand Autumns of Jacob de Zoet",
      author: "David Mitchell",
      prefix: "The grandfather clock in the State Room strikes ",
      suffix: " times.",
    },
    {
      time: "ten o'clock",
      book: "A crime in the neighborhood",
      author: "Suzanne Berne",
      prefix:
        "The light in Mr. Green's kitchen snapped off at nine-thirty, followed by the light in his bedroom at his usual ",
      suffix: ". His house was the first on the street to go dark.",
    },
    {
      time: "ten o'clock",
      book: "On Chesil Beach",
      author: "Ian McEwan",
      prefix:
        "They were alone then, and theoretically free to do whatever they wanted, but they went on eating the dinner they had no appetite for. Florence set down her knife and reached for Edward's hand and squeezed. From downstairs they heard the wireless, the chimes of Big Ben at the start of the ",
      suffix: " news.",
    },
    {
      time: "ten o'clock",
      book: "Anne Frank: The Diary of a Young Girl",
      author: "Anne Frank",
      prefix:
        "We let our upstairs room to a certain Mr. Goudsmit, a divorced man in his thirties, who appeared to have nothing to do on this particular evening; we simply could not get rid of him without being rude; he hung about until ",
      suffix: ".",
    },
  ],
  "22:02": [
    {
      time: "10.02pm.",
      book: "The Lost Symbol",
      author: "Dan Brown",
      prefix: "It was now ",
      suffix: " He has less than two hours.",
    },
  ],
  "22:05": [
    {
      time: "10:05 p.m.",
      book: "The Year of Magical Thinking",
      author: "Joan Didion",
      prefix:
        "The A-B elevator was our elevator, the elevator on which the paramedics came up at 9:20 p.m., the elevator on which they took John (and me) downstairs to the ambulance at ",
      suffix: "",
    },
  ],
  "22:06": [
    {
      time: "after ten o'clock",
      book: "The Dead",
      author: "James Joyce",
      prefix:
        "Of course, they had good reason to be fussy on such a night. And then it was long ",
      suffix:
        " and yet there was no sign of Gabriel and his wife. Besides they were dreadfully afraid that Freddy Malins might turn up screwed.",
    },
  ],
  "22:08": [
    {
      time: "Ten eight",
      book: "The Quiet American",
      author: "Graham Greene",
      prefix:
        '"My watch is always a little fast," I said. "What time do you make it now?" "',
      suffix: '." "Ten eighteen by mine. You see."',
    },
  ],
  "22:10": [
    {
      time: "ten minutes past ten",
      book: "The Death of Olivier Becaille",
      author: "Emile Zola",
      prefix:
        "That was the past, and now I had just died on the narrow couch of a Paris lodging house, and my wife was crouching on the floor, crying bitterly. The white light before my left eye was growing dim, but I remembered the room perfectly. On the left there was a chest of drawers, on the right a mantelpiece surmounted by a damaged clock without a pendulum, the hands of which marked ",
      suffix:
        ". The window overlooked the Rue Dauphine, a long, dark street. All Paris seemed to pass below, and the noise was so great that the window shook.",
    },
    {
      time: "10.10pm.",
      book: "Bridget Jones's Diary",
      author: "Helen Fielding",
      prefix: "",
      suffix:
        " When you turn your recorder on you must adjust clock and the calendar.......Press red and nothing happens. Press numbers and nothing happens. Wish stupid video had never been invented",
    },
  ],
  "22:11": [
    {
      time: "eleven minutes past ten",
      book: "The Enemy",
      author: "Lee Child",
      prefix:
        "Therefore a sergeant called Trifonov had been on post all day or all week and then he had left at ",
      suffix: " in the evening.",
    },
  ],
  "22:12": [
    {
      time: "2212",
      book: "Infinite Jest",
      author: "David Foster Wallace",
      prefix:
        "The Chinese women scuttled at an amazing rate, given their size and the bags' size. It was c. ",
      suffix:
        ":30-40h., smack in the middle of the former Interval of Issues Resolution.",
    },
  ],
  "22:14": [
    {
      time: "2214",
      book: "Infinite Jest",
      author: "David Foster Wallace",
      prefix:
        "The shopping bags looked heavy and impressive, their weight making the Chinese women lean in slightly towards each other. Call it ",
      suffix: ":10h.",
    },
  ],
  "22:15": [
    {
      time: "10:15 p.m.",
      book: "Bridget Jones's Diary",
      author: "Helen Fielding",
      prefix: "",
      suffix: " Aargh Newsnight on in 15 minute",
    },
  ],
  "22:17": [
    {
      time: "10:17 p. m.",
      book: "Bridget Jones's Diary",
      author: "Helen Fielding",
      prefix: "",
      suffix: " Casette will not go i",
    },
  ],
  "22:18": [
    {
      time: "Ten eighteen",
      book: "The Quiet American",
      author: "Graham Greene",
      prefix:
        '"My watch is always a little fast," I said. "What time do you make it now?" "Ten eight." "',
      suffix: ' by mine. You see."',
    },
    {
      time: "10:18pm.",
      book: "Bridget Jones's Diary",
      author: "Helen Fielding",
      prefix: "",
      suffix: " Ah. Thelma and Louise is in ther",
    },
  ],
  "22:20": [
    {
      time: "10:20",
      book: "A Wild Sheep Chase",
      author: "Haruki Murakami",
      prefix: "At ",
      suffix:
        ' she returned with a shopping bag from the supermarket. In the bag were three scrub brushes, one box of paperclips and a well-chilled six-pack of canned beer. So I had another beer. "It was about sheep," I said. "Didn\'t I tell you?" she said.',
    },
  ],
  "22:21": [
    {
      time: "10:21pm.",
      book: "Bridget Jones's Diary",
      author: "Helen Fielding",
      prefix: "",
      suffix:
        " Frenziedly press all buttons. Cassette comes out and goes back in agai",
    },
    {
      time: "10:21pm.",
      book: "Bridget Jones's Diary",
      author: "Helen Fielding",
      prefix: "",
      suffix: " Thelma and Louise will not come ou",
    },
    {
      time: "2221h",
      book: "Infinite Jest",
      author: "David Foster Wallace",
      prefix: "On a Saturday c. ",
      suffix:
        "., Lenz found a miniature bird that had fallen out of some nest and was sitting bald and pencil-necked on the lawn of Unit #3 flapping ineffectually, and went in with Green and ducked Green and went back outside to # 3's lawn and put the thing in a pocket and went in and put it down the garbage disposal in the kitchen sink of the kitchen, but still felt largely impotent and unresolved.",
    },
    {
      time: "2221h",
      book: "Infinite Jest",
      author: "David Foster Wallace",
      prefix: "On a Saturday c. ",
      suffix:
        "., Lenz found a miniature bird that had fallen out of some nest and was sitting bald and pencil-necked on the lawn of Unit #3 flapping ineffectually...",
    },
  ],
  "22:24": [
    {
      time: "10:24",
      book: "A Short History of Nearly Everything",
      author: "Bill Bryson",
      prefix: "Thanks to ten minutes or so of balmy weather, by ",
      suffix:
        " the Earth is covered in the great carboniferous forests whose residues give us all our coal, and the first winged insects are evident.",
    },
  ],
  "22:25": [
    {
      time: "10:25pm.",
      book: "Bridget Jones's Diary",
      author: "Helen Fielding",
      prefix: "",
      suffix:
        ' Got new cassette in now. Right. Turn to "Recording.................. Aargh Newsnight is startin',
    },
  ],
  "22:26": [
    {
      time: "ten-twenty-six",
      book: "Hard Boiled Wonderland and the End of the World",
      author: "Haruki Murakami",
      prefix:
        "As always, consciousness returned to me progressively from the edges of my field of vision. The first things to claim recognition were the bathroom door emerging from the far right and a lamp from the far left, from which my awareness gradually drifted inward like ice flowing together toward the middle of a lake. In the exact center of my visual field was the alarm clock, hands pointing to ",
      suffix: ".",
    },
  ],
  "22:27": [
    {
      time: "twenty-seven minutes past 10",
      book: "England, Their England",
      author: "AG Macdonell",
      prefix: "Mr Harcourt woke up with mysterious suddenness at ",
      suffix:
        ", and, by a curious coincidence, it was at that very instant that the butler came in with two footmen laden with trays of whisky, brandy, syphons, glasses and biscuits.",
    },
  ],
  "22:30": [
    {
      time: "ten thirty",
      book: "Brooklyn",
      author: "Colm Toibin",
      prefix: "She looked at the clock; it was ",
      suffix:
        ". If she could get there quickly on the subway, then she could be at his house in less than an hour, maybe a bit longer if the late trains did not come so often.",
    },
    {
      time: "ten-thirty",
      book: "Smiley's People",
      author: "John Le Carre",
      prefix: "The time was ",
      suffix:
        " but it could have been three in the morning, because along its borders, West Berlin goes to bed with the dark",
    },
  ],
  "22:31": [
    {
      time: "10.31pm.",
      book: "Bridget Jones's Diary",
      author: "Helen Fielding",
      prefix: "",
      suffix:
        " Ok OK. Calm. Penny Husbands-Bosworth, so asbestos leukaemia item is not on yet",
    },
    {
      time: "10.31 pm",
      book: "The Curious Incident of the Dog in the Night-time",
      author: "Mark Haddon",
      prefix: "And, later on, at ",
      suffix:
        ", I went out onto the balcony to find out whether I could see any stars, but there weren't any because of all the clouds and what is called Light Pollution which is light from streetlights and car headlights and floodlights and lights in buildings reflecting off tiny particles in the atmosphere and getting in the way of light from the stars. So I went back inside.",
    },
  ],
  "22:33": [
    {
      time: "10:33 p.m.",
      book: "Bridget Jones's Diary",
      author: "Helen Fielding",
      prefix: "",
      suffix:
        " Yessss, yessss. RECORDING CURRENT PROGRAMME. Have done it. Aargh. All going mad. Cassette has started rewinding and now stopped and ejected. Why? Shit. Shit. Realize in excitement have sat on remote control",
    },
    {
      time: "10:33pm.",
      book: "Bridget Jones's Diary",
      author: "Helen Fielding",
      prefix: "",
      suffix:
        " Yessss, yessss. RECORDING CURRENT PROGRAMME. Have done it. Aargh. All going mad. Cassette has started rewinding and now stopped and ejected. Why? Shit. Shit. Realize in excitement have sat on remote control",
    },
  ],
  "22:35": [
    {
      time: "10:35 p.m.",
      book: "Bridget Jones's Diary",
      author: "Helen Fielding",
      prefix: "",
      suffix:
        " Frantic now. Have rung Sahzzer, Rebecca, Simon, Magda. Nobody knows how to programme their videos. Only person I know who knows how to do it is Daniel",
    },
  ],
  "22:40": [
    {
      time: "twenty to eleven",
      book: "The Man Who Watched the Trains Go By",
      author: "Georges Simenon",
      prefix: "The station clock told him the time: ",
      suffix:
        ". He went to the booking office and asked the clerk in a polite tone when was the next train to Paris. 'In twelve minutes.'",
    },
  ],
  "22:41": [
    {
      time: "10:41",
      book: "Appointment in Samarra",
      author: "John O'Hara",
      prefix:
        'He climbed into the front seat and started the car. It started with a merry powerful hum, ready to go. "There, the bastards", said Julian, and smashed the clock with the bottom of the bottle, to give them an approximate time. It was ',
      suffix: "",
    },
  ],
  "22:44": [
    {
      time: "About a quarter to eleven.",
      book: "Dead in the Water",
      author: "Carola Dunn",
      prefix:
        'Alec pricked up his ears. "When was that?" "Oh, yesterday evening." "What time?" "',
      suffix: ' I was playing bridge."',
    },
  ],
  "22:45": [
    {
      time: "10.45pm.",
      book: "Bridget Jones's Diary",
      author: "Helen Fielding",
      prefix: "",
      suffix:
        " Oh God Daniel fell about laughing when I said I could not programme video. Said he would do it for me. Still at least I have done best for Mum. It is exciting and historic when one's friends are on TV",
    },
    {
      time: "fifteen minutes before eleven",
      book: "Good Intentions",
      author: "Ogden Nash",
      prefix:
        "So the Lackadaisical Broadcasting Co. bids you farewell with the message that if you aren't grateful to be living in a world where so many things to be grateful for are yours as a matter of course. Why it is now five seconds until ",
      suffix: " o'clock and you are just an old Trojan Horse.",
    },
  ],
  "22:46": [
    {
      time: "10.46 p.m.",
      book: "The Riddle of the Sands",
      author: "Erskine Childers",
      prefix:
        "The 'night train' tallied to perfection, for high tide in the creek would be, as Davies estimated, between 10.30 and 11.00 p.m.on the night of the 25th; and the time-table showed that the only night train arriving at Norden was one from the south at ",
      suffix: "",
    },
  ],
  "22:48": [
    {
      time: "10.48",
      book: "The Sign in the Sky",
      author: "Agatha Christie",
      prefix:
        '"Oh! I don\'t know about that," said Mr. Satterthwaite, warming to his subject. "I was down there for a bit last summer. I found it quite convenient for town. Of course the trains only go every hour. 48 minutes past the hour from Waterloo-up to ',
      suffix: '."',
    },
  ],
  "22:49": [
    {
      time: "well after 2245h",
      book: "Infinite Jest",
      author: "David Foster Wallace",
      prefix: "It's ",
      suffix:
        ". The dog's leash slides hissing to the end of the Day-Glo line and stops the dog a couple of paces from the inside of the gate, where Lenz is standing, inclined in the slight forward way of somebody who's talking baby-talk to a dog.",
    },
  ],
  "22:50": [
    {
      time: "10.50 P. M.",
      book: "The Parasite",
      author: "Sir Arthur Conan Doyle",
      prefix: "",
      suffix:
        " This diary-keeping of mine is, I fancy, the outcome of that scientific habit of mind about which I wrote this morning. I like to register impressions while they are fresh",
    },
    {
      time: "ten to eleven",
      book: "Last Orders",
      author: "Graham Swift",
      prefix:
        "Saturday night. And I said, 'It's a hundred this year, ain't anybody noticed?'\"Jack said, 'What's a hundred?' I said, 'Pub is. Coach is. Look at the clock.' Jack said, ‘It's ",
      suffix: "’.",
    },
    {
      time: "22.50",
      book: "Day",
      author: "A. L. Kennedy",
      prefix:
        "So think yourself lucky while you're awake and remember a happy crew. Think of Hamburg on the Magic Night. ",
      suffix:
        " and they went out neatly, just as they should - you couldn't fault Parks, he was always on his route.",
    },
  ],
  "22:55": [
    {
      time: "Eleven o'clock, all but five minutes!",
      book: "The Phantom of the Opera",
      author: "Gaston Leroux",
      prefix: "\"It is eleven o'clock! ",
      suffix:
        '" "But which eleven o\'clock?" "The eleven o\'clock that is to decide life or death!...He told me so just before he went....He is terrible....He is quite mad: he tore off his mask and his yellow eyes shot flames!..."',
    },
  ],
  "22:58": [
    {
      time: "just about eleven o’clock",
      book: "Wuthering Heights",
      author: "Emily Brontë",
      prefix:
        "Then it grew dark; she would have had them to bed, but they begged sadly to be allowed to stay up; and, ",
      suffix: ", the door-latch was raised quietly, and in stepped the master.",
    },
  ],
  "22:59": [
    {
      time: "one minute to eleven",
      book: "The Complaints",
      author: "Ian Rankin",
      prefix: "They parked the car outside Lowther's at precisely ",
      suffix:
        ". People were leaving, not all of them happy at having their evening curtailed. But the grumbling was muted, and even then it only started once they were safely on the street.",
    },
  ],
  "23:00": [
    {
      time: "at eleven",
      book: "Smiley's People",
      author: "John le Carre",
      prefix: "'He will be here ",
      suffix: " exactly, sir.' At the bar, naked couples had begun dancing.",
    },
    {
      time: "eleven o'clock",
      book: "Tess of the d'Urbervilles",
      author: "Thomas Hardy",
      prefix: "At ",
      suffix:
        " that night, having secured a bed at one of the hotels and telegraphed his address to his father immediately on his arrival, he walked out into the streets of Sandbourne.",
    },
    {
      time: "eleven o'clock",
      book: "The Moonstone",
      author: "Wilkie Collins",
      prefix: "At ",
      suffix:
        ", I rang the bell for Betteredge, and told Mr. Blake that he might at last prepare himself for bed.",
    },
    {
      time: "eleven o'clock",
      book: "If on a winter's night a traveller",
      author: "Italo Calvino",
      prefix:
        'He says, "They\'ve killed Jan. Clear out." "The suitcase?" I ask. "Take it away again. We want nothing to do with it now. Catch the ',
      suffix:
        ' express." "But it doesn\'t stop here...." "It will. Go to track six. Opposite the freight station. You have three minutes." "But..." "Move, or I\'ll have to arrest you."',
    },
    {
      time: "eleven",
      book: "Jane Eyre",
      author: "Charlotte Brontë",
      prefix: "The clock struck ",
      suffix:
        ". I looked at Adele, whose head leant against my shoulder; her eyes were waxing heavy, so I took her up in my arms and carried her off to bed. It was near one before the gentlemen and ladies sought their chambers.",
    },
    {
      time: "eleven",
      book: "Jane Eyre",
      author: "Charlotte Brontë",
      prefix: "The clock struck ",
      suffix:
        ". I looked at Adele, whose head leant against my shoulder; her eyes were waxing heavy, so I took her up in my arms and carried her off to bed. It was near one before the gentlemen and ladies sought their chambers.",
    },
    {
      time: "at eleven",
      book: "The Recognitions",
      author: "William Gaddis",
      prefix: "The train arrived in New York ",
      suffix: " that night.",
    },
    {
      time: "2300h",
      book: "Infinite Jest",
      author: "David Foster Wallace",
      prefix: "They didn't even sit down to eat until ",
      suffix: ".",
    },
    {
      time: "eleven o'clock",
      book: "Harry Potter and the Order of the Phoenix",
      author: "JK Rowling",
      prefix: "When they reached the top of the Astronomy Tower at ",
      suffix:
        ", they found a perfect night for stargazing, cloudless and still.",
    },
  ],
  "23:03": [
    {
      time: "Eleven oh-three",
      book: "Little Green Men",
      author: "Christopher Buckley",
      prefix:
        '"What makes you think it\'s for real?" "Just a hunch, really. He sounded for real. Sometimes you can just tell about people"-he smiled-"even if you\'re a dull old WASP." "I think it\'s a setup." "Why?" "I just do. Why would someone from the government want to help you?" "Good question. Guess I\'ll find out." She went back into the kitchen."What time are you meeting him?" she called out. "',
      suffix:
        '," he said. "That made me think he\'s for real. Military and intelligence types set precise appointment times to eliminate confusion and ambiguity. Nothing ambiguous about eleven oh-three."',
    },
  ],
  "23:05": [
    {
      time: "11.05",
      book: "The Go-Between",
      author: "L.P.Hartley",
      prefix: "It was ",
      suffix:
        ", five minutes later than my habitual bedtime. I felt. I felt guilty at being still up, but the past kept pricking at me and I knew that all the events of those nineteen days in July were astir within me, like the loosening phlegm in an attack of bronchitis",
    },
    {
      time: "five minutes past eleven",
      book: "The Poison Belt",
      author: "Sir Arthur Conan Doyle",
      prefix: "It was ",
      suffix:
        " when I made my last entry. I remember winding up my watch and noting the time. So I have wasted some five hours of the little span still left to us. Who would have believed it possible? But I feel very much fresher, and ready for my fate--or try to persuade myself that I am. And yet, the fitter a man is, and the higher his tide of life, the more must he shrink from death. How wise and how merciful is that provision of nature by which his earthly anchor is usually loosened by many little imperceptible tugs, until his consciousness has drifted out of its untenable earthly harbor into the great sea beyond!",
    },
    {
      time: "11:05",
      book: "Household Worms",
      author: "Stanley Donwood",
      prefix: "My watch says ",
      suffix: ". But whether AM or PM I don't know.",
    },
  ],
  "23:07": [
    {
      time: "11.07 pm",
      book: "The Andromeda Strain",
      author: "Michael Crichton",
      prefix: "At ",
      suffix:
        ', Samuel "Gunner" Wilson was moving at 645 miles per hour over the Mojave Desert. Up ahead in the moonlinght, he saw the twin lead jets, their afterburners glowing angrily in the night sky.',
    },
  ],
  "23:10": [
    {
      time: "ten past eleven",
      book: "Deaf Sentence",
      author: "David Lodge",
      prefix: "Another Christmas day is nearly over. It's ",
      suffix:
        ". Richard declined with thanks my offer to make up a bed for him here in my study, and has driven off back to Cambridge, so I am able to make some notes on the day before going to bed myself.",
    },
    {
      time: "ten minutes past eleven",
      book: "Appointment in Samarra",
      author: "John O'Hara",
      prefix: "He had not the strength to help himself, and at ",
      suffix: " no one could have helped him, no one in the world",
    },
  ],
  "23:11": [
    {
      time: "11:11 p.m.",
      book: "The Year of Magical Thinking",
      author: "Joan Didion",
      prefix:
        'Life changes fast Life changes in an instant You sit down to dinner and life as you know it ends. The Question of self-pity. Those were the first words I wrote after it happened. The computer dating on the Microsoft Word file ("Notes on change.doc") reads "May 20, 2004, ',
      suffix:
        '," but that would have been a case of my opening the file and reflexively pressing save when I closed it. I had made no changes to that file since I wrote the words, in January 2004, a day or two after the fact. For a long time I wrote nothing else. Life changes in the instant. The ordinary instant.',
    },
  ],
  "23:12": [
    {
      time: "23:12",
      book: "A Naked Singularity",
      author: "Sergio De La Pava",
      prefix:
        "There was a confirmatory identification done by undercover officer 6475 at ",
      suffix: " hours at the corner of 147th and Amsterdam.",
    },
  ],
  "23:15": [
    {
      time: "11.15pm.",
      book: "Bridget Jones's Diary",
      author: "Helen Fielding",
      prefix: "",
      suffix:
        " Humph. Mum just rang \"Sorry, darling. It isn't Newsnigtht, it's Breakfast News tomorrow. Could you set it for seven o'clock tomorrow morning, BBC1?",
    },
    {
      time: "quarter-past eleven",
      book: "Dairy of a Nobody",
      author: "George and Weedon Grossmith",
      prefix: "On arriving home at a ",
      suffix:
        ", we found a hansom cab, which had been waiting for me for two hours with a letter. Sarah said she did not know what to do, as we had not left the address where we had gone.",
    },
  ],
  "23:16": [
    {
      time: "11.16 pm.",
      book: "The Curious Incident of the Dog in the Night-time",
      author: "Mark Haddon",
      prefix:
        "But I couldn't get out of the house straight away because he would see me, so I would have to wait until he was asleep. The time was ",
      suffix:
        " I tried doubling 2s again, but I couldn't get past 2(15) which was 32,768. So I groaned to make the time pass quicker and not think.",
    },
  ],
  "23:18": [
    {
      time: "11.18",
      book: "Trumpet",
      author: "Jackie Kay",
      prefix: "It is ",
      suffix:
        ". A row of bungalows in a round with a clump of larch tree in the middle.",
    },
  ],
  "23:19": [
    {
      time: "11:19",
      book: "Blackout",
      author: "Connie Willis",
      prefix:
        "A whistle cut sharply across his words. Peter got onto his knees to look out the window, and Miss Fuller glared at him. Polly looked down at her watch: ",
      suffix: ". The train. But the stationmaster had said it was always late.",
    },
  ],
  "23:20": [
    {
      time: "eleven-twenty",
      book: "Watchers",
      author: "Dean Koontz",
      prefix: "From Balboa Island, he drove south to Laguna Beach. At ",
      suffix: ", he parked his van across the street from the Hudston house.",
    },
    {
      time: "twenty past eleven",
      book: "Captains Courageous",
      author: "Rudyard Kipling",
      prefix: "Harvey looked at the clock, which marked ",
      suffix:
        ". \"Then I'll sleep here till three and catch the four o'clock freight. They let us men from the Fleet ride free as a rule.\"",
    },
  ],
  "23:22": [
    {
      time: "11.22",
      book: "Dreams of leaving",
      author: "Robert Thomson",
      prefix: "At ",
      suffix:
        " he handed his ticket to a yawning guard and walked down a long flight of wooden steps to the car-park. A breeze lifted and dropped the leaves of a tree, and he thought of the girl with the blonde hair. His bicycle lay where he had left it.",
    },
  ],
  "23:25": [
    {
      time: "11.25 p.m.",
      book: "Other People's Money",
      author: "Justin Cartwright",
      prefix: '"OK, Estelle, I willl be at Nice Airport at ',
      suffix: ' on Saturday, BA: Could you send the driver?"',
    },
    {
      time: "eleven o'clock and twenty-five minutes",
      book: "A Wireless Message",
      author: "Ambrose Bierce",
      prefix:
        "To test the intensity of the light whose nature and cause he could not determine, he took out his watch to see if he could make out the figures on the dial. They were plainly visible, and the hands indicated the hour of ",
      suffix:
        ". At that moment the mysterious illumination suddenly flared to an intense, an almost blinding splendor…",
    },
  ],
  "23:26": [
    {
      time: "11:26 p.m.",
      book: "American Gods",
      author: "Neil Gaiman",
      prefix: "Los Angeles. ",
      suffix:
        " In a dark red room- the color of the walls is close to that of raw liver- is a tall woman dressed cartoonishly in too-tight silk shorts, her breasts pulled up and pressed forward by the yellow blouse tied beneath them.",
    },
  ],
  "23:27": [
    {
      time: "23.27",
      book: "Other People's Money",
      author: "Justin Cartwright",
      prefix:
        "He tells his old friend the train times and they settle on the 19.45 arriving at ",
      suffix:
        '. "I\'ll book us into the ultra-luxurious Francis Drake Lodge. Running water in several rooms. Have you got a mobile?"',
    },
  ],
  "23:30": [
    {
      time: "2330",
      book: "Infinite Jest",
      author: "David Foster Wallace",
      prefix:
        "He loaded the player and turned on the viewer, his knees popping again as he squatted to set the cue to ",
      suffix: ".",
    },
    {
      time: "half past eleven",
      book: "'The Dismissal', in Midnight Mass & Other Stories",
      author: "Paul Bowles",
      prefix:
        "He would catch the night bus for Casablanca, the one that left the beach at ",
      suffix: ".",
    },
    {
      time: "half-past eleven",
      book: "“The Voyage”",
      author: "Katherine Mansfield",
      prefix: "The Picton boat was due to leave at ",
      suffix:
        ". It was a beautiful night, mild, starry, only when they got out of the cab and started to walk down the Old Wharf that jutted out into the harbour, a faint wind blowing off the water ruffled under Fenella's hat, and she put up her hand to keep it on.",
    },
    {
      time: "half past eleven",
      book: "Under Milk Wood",
      author: "Dylan Thomas",
      prefix: "The ship's clock in the bar says ",
      suffix:
        ". Half past eleven is opening time. The hands of the clock have stayed still at half past eleven for fifty years.",
    },
  ],
  "23:31": [
    {
      time: "twenty-nine minutes to midnight",
      book: "Midnight's Children",
      author: "Salman Rushdie",
      prefix: "It is ",
      suffix:
        ". Dr Narlikar's Nursing Home is running on a skeleton staff; there are many absentees, many employees who have preferred to celebrate the imminent birth of the nation, and will not assist tonight at the births of children.",
    },
  ],
  "23:32": [
    {
      time: "In about twenty-eight minutes it will be midnight.",
      book: "The Evenings",
      author: "Gerard Reve",
      prefix: '"This is the evening. This is the night. It is New Year´s Eve. ',
      suffix:
        ' I still have twenty-eight minutes left. I have to recollect my thoughts. At twelve o´clock, I should be done thinking." He looked at his father. "Help those that are depressed and consider themselves lost in this world," he thought. "Old fart."',
    },
    {
      time: "11.32 pm",
      book: "The Curious Incident of the Dog in the Night-time",
      author: "Mark Haddon",
      prefix:
        "And then it started to rain and I got wet and I started shivering because I was cold. And then it was ",
      suffix:
        " and I heard voices of people walking along the street. And a voice said, 'I don't care whether you thought it was funny or not,' and it was a lady's voice.",
    },
  ],
  "23:33": [
    {
      time: "twenty-seven minutes to midnight",
      book: "Midnight's Children",
      author: "Salman Rushdie",
      prefix:
        "We are on Colaba Causeway now, just for a moment, to reveal that at ",
      suffix:
        ", the police are hunting for a dangerous criminal. His name: Joseph D'Costa. The orderly is absent, has been absent for several days, from his work at the Nursing Home, from his room near the slaughterhouse, and from the life of a distraught virginal Mary",
    },
  ],
  "23:34": [
    {
      time: "Eleven thirty-four",
      book: "American Psycho",
      author: "Bret Easton Ellis",
      prefix: "",
      suffix:
        ". We stand on the sidewalk in front of Jean's apartment on the Upper East Side. Her doorman eyes us warily and fills me with a nameless dread, his gaze piercing me from the lobby. A curtain of stars, miles of them, are scattered, glowing, across the sky and their multitude humbles me, which I have a hard time tolerating. She shrugs and nods after I say something about forms of anxiety. It's as if her mind is having a hard time communicating with her mouth, as if she is searching for a rational analysis of who I am, which is, of course, an impossibility: there ... is ... no ... key",
    },
    {
      time: "eleven thirty-four",
      book: "The Hard Way",
      author: "Lee Child",
      prefix: "Reacher retrieved his G36 from under the saloon bar window at ",
      suffix:
        " precisely and set out to walk back on the road, which he figured would make the return trip faster.",
    },
  ],
  "23:35": [
    {
      time: "eleven thirty-five",
      book: "Hamlet, Revenge!",
      author: "Michael Innes",
      prefix: "Then at ",
      suffix:
        " the door at the rear of the hall opened and a police sergeant and three constables entered, ushered by Bagot.",
    },
  ],
  "23:36": [
    {
      time: "2336",
      book: "Infinite Jest",
      author: "David Foster Wallace",
      prefix: "Then Green knocks at the front door at ",
      suffix:
        " - Gately has to Log the exact time and then it's his call whether to unlock the door.",
    },
  ],
  "23:39": [
    {
      time: "11.39",
      book: "Old Possum's Book of Practical Cats",
      author: "T S Eliot",
      prefix: "There's a whisper down the line at ",
      suffix:
        " When the Night Mail's ready to depart, Saying \"Skimble where is Skimble has he gone to hunt the thimble? We must find him or the train can't start.\"",
    },
  ],
  "23:40": [
    {
      time: "11:40",
      book: "Dracula",
      author: "Bram Stoker",
      prefix:
        "We all have the maps and appliances of various kinds that can be had. Professor Van Helsing and I are to leave by the ",
      suffix:
        " train tonight for Veresti, where we are to get a carriage to drive to the Borgo Pass. We are bringing a good deal of ready money, as we are to buy a carriage and horses.",
    },
  ],
  "23:41": [
    {
      time: "11:41",
      book: "Noble House",
      author: "James Clavell",
      prefix:
        "In a little while his mind cleared, but his head ached, arms ached, body ached. The phosphorescent figures on his watch attracted his attention. He peered at them. The time was ",
      suffix: ". I remember...what do I remember?",
    },
  ],
  "23:42": [
    {
      time: "11.42",
      book: "Old Possum's Book of Practical Cats",
      author: "TS Eliot",
      prefix: "At ",
      suffix:
        " then the signal's nearly due And the passengers are frantic to a man- Then Skimble will appear and he'll saunter to the rear:",
    },
  ],
  "23:43": [
    {
      time: "eleven forty-three",
      book: "Talking to Strange Men",
      author: "Ruth Rendell",
      prefix: "The clock told him it was ",
      suffix:
        " and in that moment, in a flash of illumination, Mungo understood what the numbers at the end of Moscow Centre's messages were",
    },
  ],
  "23:44": [
    {
      time: "eleven forty-four",
      book: "Dead Famous",
      author: "Ben Elton",
      prefix: "'At ",
      suffix:
        " last night somebody stabbed this girl in the neck with a kitchen knife and immediately thereafter plunged the same knife through her skull, where it remained.’",
    },
  ],
  "23:45": [
    {
      time: "three quarters past eleven",
      book: "Oliver Twist",
      author: "Charles Dickens",
      prefix: "The church clocks chimed ",
      suffix:
        ", as two figures emerged on London Bridge. One, which advanced with a swift and rapid step, was that of a woman who looked eagerly about her as though in quest of some expected object; the other figure was that of a man...",
    },
    {
      time: "quarter to twelve",
      book: "Three Men in a Boat",
      author: "Jerome K. Jerome",
      prefix:
        "We struck the tow-path at length, and that made us happy because prior to this we had not been sure whether we were walking towards the river or away from it, and when you are tired and want to go to bed, uncertainties like that worry you. We passed Shiplake as the clock was striking the ",
      suffix:
        " and then George said thoughtfully: 'You don't happen to remember which of the islands it was, do you?'",
    },
  ],
  "23:46": [
    {
      time: "11:46 p.m.",
      book: "Doll: A Romance of the Mississippi",
      author: "Joyce Carol Oates",
      prefix:
        "In the Kismet Lounge, Mr. Early sees suddenly to his horror it's ",
      suffix:
        " He's been in this place far longer than he'd planned, and he's had more to drink than he'd planned. Shame! What if, back at the E-Z, his little girl is crying piteously for him?",
    },
  ],
  "23:47": [
    {
      time: "thirteen minutes to midnight",
      book: "Dead Soldiers Don't Sing",
      author: "Rudolf Jašík",
      prefix: "If he had glanced at his watch, he would have seen that it was ",
      suffix:
        ". And if he had been interested in what was going on, he would have heard the voices and bawling of terrified men.",
    },
  ],
  "23:48": [
    {
      time: "11.48pm.",
      book: "American Tabloid",
      author: "James Ellroy",
      prefix:
        "Littell arranged a private charter.He told the pilot to fly balls-to-the-wall.The little two-seater rattled and shook-Kemper couldn't believe it. It was ",
      suffix: " They were thirty-six hours short of GO.",
    },
  ],
  "23:49": [
    {
      time: "eleven minutes to midnight",
      book: "The Boy Who Followed Ripley",
      author: "The Patricia Highsmith",
      prefix:
        "Tom shrugged. He pushed his pinkish ruffled sleeve back, and saw that it was ",
      suffix: ". Tom finished his coffee.",
    },
  ],
  "23:50": [
    {
      time: "11.50pm",
      book: "Extremely Loud and Incredibly Close",
      author: "Jonathan Safran Foer",
      prefix: "At ",
      suffix:
        ", I got up extremely quietly, took my things from under the bed, and opened the door one millimeter at a time, so it wouldn't make any noise.",
    },
  ],
  "23:51": [
    {
      time: "eleven-fifty-one",
      book: "The Mystery of Dr Fu Manchu",
      author: "Sax Rohmer",
      prefix: '"Due at Waterloo at ',
      suffix:
        '," panted Smith."That gives us thirty-nine minutes to get to the other side of the river and reach his hotel."',
    },
  ],
  "23:52": [
    {
      time: "eight minutes to midnight",
      book: "The Green Man",
      author: "Kingsley Amis",
      prefix: "It was ",
      suffix:
        ". Just nice time, I said to myself. Indoors, everything was quiet and in darkness. Splendid. I went to the bar and fetched a tumbler, a siphon of soda and a bottle of Glen Grant, took a weak drink and a pill, and settled down in the public dining-room to wait the remaining two minutes.",
    },
  ],
  "23:53": [
    {
      time: "7 minutes to midnight",
      book: "The Curious Incident of the Dog in the Night-time",
      author: "Mark Haddon",
      prefix: "It was ",
      suffix:
        ". The dog was lying on the grass in the middle of the lawn in front of Mrs. Shears' house.",
    },
  ],
  "23:54": [
    {
      time: "11.54pm",
      book: "Conclave",
      author: "Greg Tobin",
      prefix: "His watch read ",
      suffix:
        " Eastern Standard Time. Already it was nearly 6.00am in Rome. He had left a city frozen by a harsh January storm, after a bleak, wet Christmas season.",
    },
  ],
  "23:55": [
    {
      time: "five minutes to midnight",
      book: "Harry Potter and the Prisoner of Azkaban",
      author: "J. K. Rowling",
      prefix: '"I am going to lock you in. It is-" he consulted his watch, "',
      suffix: '. Miss Granger, three turns should do it. Good luck."',
    },
    {
      time: "five minutes to twelve",
      book: "The Moonstone",
      author: "Wilkie Collins",
      prefix: "I looked at my watch. It wanted ",
      suffix:
        ", when the premonitory symptoms of the working of the laudanum first showed themselves to me. At this time, no unpractised eyes would have detected any change in him. But, as the minutes of the new morning wore away, the swiftly-subtle progress of the influence began to show itself more plainly. The sublime intoxication of opium gleamed in his eyes; the dew of a stealthy perspiration began to glisten on his face. In five minutes more, the talk which he still kept up with me, failed in coherence.",
    },
  ],
  "23:56": [
    {
      time: "four minutes to midnight",
      book: "Wicked Women",
      author: "Fay Weldon",
      prefix:
        "The human race is at the end of the line, the doomsday clock ticks on. It's stopped for a decade at ",
      suffix:
        ", but there the hands still stand. Any minute now they'll begin to move again.",
    },
  ],
  "23:57": [
    {
      time: "Eleven fifty-seven",
      book: "No Country for Old Men",
      author: "Cormac McCarthy",
      prefix:
        'Wells looked out at the street. "What time is it?" he said. Chigurh raised his wrist and looked at his watch. "',
      suffix:
        "\" he said. Wells nodded. By the old woman's calendar I've got three more minutes.",
    },
  ],
  "23:58": [
    {
      time: "one minute and seventeen seconds before midnight",
      book: "A Short History of Nearly Everything",
      author: "Bill Bryson",
      prefix: "Humans emerge ",
      suffix:
        ". The whole of our recorded history, on this scale, would be no more than a few seconds, a single human lifetime barely an instant.",
    },
  ],
  "23:59": [
    {
      time: "a minute to midnight",
      book: "The Man Who Walked Through Walls",
      author: "Marcel Aymé",
      prefix: "At ",
      suffix:
        ", Roquenton was holding his wife's hand and giving her some last words of advice. On the stroke of midnight, she felt her companion's hand melt away inside her own.",
    },
    {
      time: "new day was still a minute away",
      book: "No Country for Old Men",
      author: "Cormac McCarthy",
      prefix:
        "Chigurgh rose and picked up the empty casing off the rug and blew into it and put it in his pocket and looked at his watch. The ",
      suffix: ".",
    },
  ],
};

export default quotes;
