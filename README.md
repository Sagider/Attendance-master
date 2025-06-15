# Attendance-master
This is an app that can be used to take the attendance of an entire class by simply taking a picture of the class with everyone in it

All the students/individuals who need to be identified must have their images stored in a directory called "known_faces" under a sub directory which has the same name as the person.
These images must have only the individual's face in it, the presence of more tham one face will not allow the model to identify the person while training.

It is recommended to have the images to be identified in a directory called "unkown_faces" at root level, but as long as path is specified, any accesible location is acceptable.
