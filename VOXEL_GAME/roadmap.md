#

# iter 3

great. so now we have the "place block" and "place block glued to face" mechanics.



i want you to replace the "destroy block" mechanic with two dfferent, but similar mechanics. design these (how do they work, what keybindings, etc) and implement them.

* "destroy up structure aiming at" (a structure is a collection of blocks that are glued together, e.g. a pillar consisting of three blocks glued on top of each other)

  * if that structure is too big (choose some threshold), dont destroy it
* "unglue block from neighbour"

  * unglues the block from a neighbouring block, if they were glued. only a single glued face is un-glued (not all faces of a block at once)
  * design it so that the face that is unglued is determined by how the player is aiming at the block. maybe some key combination? or a tool like the "ungluer" similar to e.g. shears in minecraft





make the floor wider. it should consist of a single layer of gravel. all unglued. normally gravel like this would fall but add a hard-coded "floor" where blocks dont fall below

