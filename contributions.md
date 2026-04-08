# Contributions Summary

## Gabriel

## Gordon

## Sean

Implemented base rock paper scissors game to be used for simulating matches in environment.

## Russell

# Contributions Table

| Contribution           | Person  | Description                                                                                                                                                                               |
| ---------------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Base RPS Game          | Sean    | Implemented base rock paper scissors game. Added example of manual game (manual_rps_example.py).                                                                                          |
| Designed Player Class  | Gordon  | Designed and implemented player class featuring the necessary fields and methods to simulate and manage a Player in a simulated RRPS tournament                                                        |
| Naive Environemnt      | Gordon  | Created first rendition of simulated environment to simulate a complete RRPS tournament with pseudo randomized player decisions                                                                        |
| Env to Gymnasium       | Gordon  | Refactored the evnvironment to utilize the gymansium library, and completely interactable through the Step() method                                                                                             |
| Refactor Env           | Russell | refactored enviorment to be usable as a library                                                                                                                                           |
| Added Movement         | Russell | updated rps to include movement in the action space, update player to store position, enabling agents to now have the option. Players now can only challenge players in a 1 tile distance |
| Added Visualizer       | Russell | Added vizualiser with pygame to allow playing rrps and viewing the agents in the environment                                                                                              |
| Added Basic Hashing    | Russell | Added first attempt at a hashing function for Q-learning                                                                                                                                  |
| Added Basic Q-learning | Russell | Added first attempt at Q learning implementation achieved 10 average reward                                                                                                               |
