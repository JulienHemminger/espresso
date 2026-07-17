from ursina import *
from ursina.prefabs.first_person_controller import FirstPersonController

# Initialize the Ursina app
app = Ursina()

# Enable basic shadows
import ursina.shaders

Entity.default_shader = ursina.shaders.lit_with_shadows_shader

# Hide default UI elements
window.fps_counter.enabled = False
window.exit_button.enabled = False

# Game variables
active_block_type = 0
block_types = ["gravel", "sand"]
blocks_in_world = []
disjoint_lines = []
is_paused = False

# Define block scale (8x smaller than original size 1.0)
BLOCK_SCALE = 0.125


class Block(Button):
    def __init__(self, position=(0, 0, 0), block_type="gravel", glued=False):
        block_color = color.dark_gray if block_type == "gravel" else color.yellow

        super().__init__(
            parent=scene,
            position=position,
            model="cube",
            scale=BLOCK_SCALE,
            origin_y=0.5,
            texture=None,
            color=block_color,
            highlight_color=block_color.tint(0.15),
        )
        self.block_type = block_type
        self.glued = glued
        blocks_in_world.append(self)

        # Check disjoint lines shortly after spawning
        invoke(update_disjoint_lines, delay=0.02)

    def input(self, key):
        global active_block_type, is_paused
        if is_paused:
            return  # Ignore gameplay interactions when paused

        if self.hovered:
            if key == "left mouse down":
                if self in blocks_in_world:
                    blocks_in_world.remove(self)
                destroy(self)
                invoke(update_disjoint_lines, delay=0.02)

            if key == "right mouse down":
                # Calculate placement position relative to the block's scale
                new_position = self.position + (mouse.normal * BLOCK_SCALE)
                is_glued = held_keys["control"]

                Block(
                    position=new_position,
                    block_type=block_types[active_block_type],
                    glued=is_glued,
                )

    def update(self):
        if self.glued or is_paused:
            return

        # Check directly below the block using the adjusted block scale
        target_below = self.position + Vec3(0, -BLOCK_SCALE, 0)

        is_blocked = False
        for b in blocks_in_world:
            # Simple margin check to avoid floating-point errors
            if b != self and distance(b.position, target_below) < (BLOCK_SCALE * 0.1):
                is_blocked = True
                break

        # Stop falling at y=0 (ground level)
        if self.y > 0.001 and not is_blocked:
            self.y -= BLOCK_SCALE
            invoke(update_disjoint_lines, delay=0.01)


def update_disjoint_lines():
    """Clears old disjoint lines and draws new ones between unglued adjacent blocks."""
    global disjoint_lines

    for line in disjoint_lines:
        destroy(line)
    disjoint_lines.clear()

    pos_map = {tuple(round(val, 4) for val in b.position): b for b in blocks_in_world}

    # Offsets adjusted to the block scale
    directions = [
        Vec3(BLOCK_SCALE, 0, 0),
        Vec3(-BLOCK_SCALE, 0, 0),
        Vec3(0, BLOCK_SCALE, 0),
        Vec3(0, -BLOCK_SCALE, 0),
        Vec3(0, 0, BLOCK_SCALE),
        Vec3(0, 0, -BLOCK_SCALE),
    ]

    processed_pairs = set()

    for pos, block in pos_map.items():
        for d in directions:
            neighbor_pos = tuple(round(val, 4) for val in (Vec3(*pos) + d))
            if neighbor_pos in pos_map:
                neighbor = pos_map[neighbor_pos]

                pair_id = tuple(sorted([id(block), id(neighbor)]))
                if pair_id in processed_pairs:
                    continue
                processed_pairs.add(pair_id)

                if not (block.glued or neighbor.glued):
                    draw_boundary_line(block.position, neighbor.position, d)


def draw_boundary_line(pos1, pos2, direction):
    """Draws a thin black wireframe square outline on the shared face boundary."""
    # Since origin_y=0.5 shifts the visual center down by half a scale height,
    # we calculate the visual center (visual_pos) to align our math
    visual_pos1 = pos1 - Vec3(0, BLOCK_SCALE / 2, 0)
    visual_pos2 = pos2 - Vec3(0, BLOCK_SCALE / 2, 0)
    midpoint = (visual_pos1 + visual_pos2) / 2

    # Scale offset slightly larger than half-size to avoid z-fighting
    offset = (BLOCK_SCALE / 2) * 1.01

    vertices = []
    if direction.x != 0:  # Left/Right boundary
        vertices = [
            midpoint + Vec3(0, -offset, -offset),
            midpoint + Vec3(0, offset, -offset),
            midpoint + Vec3(0, offset, offset),
            midpoint + Vec3(0, -offset, offset),
            midpoint + Vec3(0, -offset, -offset),
        ]
    elif direction.y != 0:  # Top/Bottom boundary
        vertices = [
            midpoint + Vec3(-offset, 0, -offset),
            midpoint + Vec3(offset, 0, -offset),
            midpoint + Vec3(offset, 0, offset),
            midpoint + Vec3(-offset, 0, offset),
            midpoint + Vec3(-offset, 0, -offset),
        ]
    elif direction.z != 0:  # Front/Back boundary
        vertices = [
            midpoint + Vec3(-offset, -offset, 0),
            midpoint + Vec3(offset, -offset, 0),
            midpoint + Vec3(offset, offset, 0),
            midpoint + Vec3(-offset, offset, 0),
            midpoint + Vec3(-offset, -offset, 0),
        ]

    line = Entity(
        model=Mesh(vertices=vertices, mode="line", thickness=2),
        color=color.black,
        lit=False,
    )
    disjoint_lines.append(line)


# --- Scene Construction ---

# Generate flat Gravel plain (15x15 grid) first so the ground is initialized
for z in range(15):
    for x in range(15):
        Block(
            position=(x * BLOCK_SCALE, 0, z * BLOCK_SCALE),
            block_type="gravel",
            glued=True,
        )

# Spawn Player slightly higher up to prevent falling through the scaled floor
player = FirstPersonController()
player.y = 50
player.cursor.enabled = False  # Disable default mouse cursor icon

# Add a clean 2D UI crosshair
crosshair = Entity(
    parent=camera.ui,
    model="quad",
    scale=(0.015, 0.015),
    texture="crosshair",
    color=color.black,
)

# Bland blue sky box
sky = Sky(color=color.cyan)

# Visual HUD indicator
selection_text = Text(
    text=f"Selected: {block_types[active_block_type].upper()}",
    position=(-0.1, 0.4),
    scale=2,
    color=color.black,
)

# Set up basic scene lighting and shadow mapping
sun = DirectionalLight(y=10, rotation=(45, -45, 0))
sun.shadow_map_resolution = (2048, 2048)

# --- Global Inputs & Pause System ---

# Tell this entity to continue processing input even when the game is paused
pause_manager = Entity(ignore_paused=True)


def pause_input(key):
    global active_block_type, is_paused

    if key == "escape":
        is_paused = not is_paused
        application.paused = is_paused

        if is_paused:
            # Release mouse, show cursor, disable controls, hide crosshair
            mouse.locked = False
            mouse.visible = True
            player.enabled = False
            crosshair.enabled = False
            selection_text.text = "PAUSED (ESC to Resume)"
        else:
            # Lock mouse, hide cursor, enable controls, show crosshair
            mouse.locked = True
            mouse.visible = False
            player.enabled = True
            crosshair.enabled = True
            selection_text.text = f"Selected: {block_types[active_block_type].upper()}"

    if not is_paused:
        if key == "scroll up":
            active_block_type = (active_block_type + 1) % len(block_types)
            selection_text.text = f"Selected: {block_types[active_block_type].upper()}"

        elif key == "scroll down":
            active_block_type = (active_block_type - 1) % len(block_types)
            selection_text.text = f"Selected: {block_types[active_block_type].upper()}"


pause_manager.input = pause_input

app.run()
