# mnkybx_pBART.py
# eemeric2@jhu.edu
# erik.emeric@gmail.com

import pygame
import random
import math
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import os
from scipy import stats
import sqlite3

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 60
GRAY_BG = (128, 128, 128)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Token colors - each color represents a specific token count (1-10)
TOKEN_COLORS = [
    (255, 0, 0),      # 1 token - Red
    (255, 127, 0),    # 2 tokens - Orange
    (255, 255, 0),    # 3 tokens - Yellow
    (0, 255, 0),      # 4 tokens - Green
    (0, 0, 255),      # 5 tokens - Blue
    (75, 0, 130),     # 6 tokens - Indigo
    (148, 0, 211),    # 7 tokens - Violet
    (255, 192, 203),  # 8 tokens - Pink
    (165, 42, 42),    # 9 tokens - Brown
    (225, 225, 225),  # 10 tokens - Very Light Gray
]

class GameState(Enum):
    # TODO: ADD SHUFFLE SCREEN TO INDICATE THAT THE token_weights have changed."SHUFFLING" is displayed on a blank background
    WELCOME = 0
    USERNAME_INPUT = 1 
    WAITING_FOR_CHOICE = 2
    REVEALING_OUTCOME = 3
    POST_REVEAL_DELAY = 4
    UPDATING_ANNULUS = 5
    BUST = 6
    WIN = 7
    PAUSED = 8
    INTER_TRIAL_DELAY = 9
    INTER_SEQUENCE_DELAY = 10

@dataclass
class Trial:
    """Represents a single trial in a sequence"""
    sequence_number: int = 0
    trial_number: int = 0
    earned_tokens: int = 0
    tokens_this_hit: int = 0
    choice: str = ""  # "HIT" or "STAY"
    result: str = ""  # "WIN", "BUST", "CONTINUE"
    total_accumulated: int = 0
    timestamp: str = ""
    reaction_time_ms: float = 0.0
    
    # NEW: GameState timestamps (in milliseconds from session start)
    timestamp_waiting_for_choice: float = 0.0
    timestamp_revealing_outcome: float = 0.0
    timestamp_post_reveal_delay: float = 0.0
    timestamp_updating_annulus: float = 0.0
    timestamp_bust: float = 0.0
    timestamp_win: float = 0.0
    timestamp_paused: float = 0.0
    timestamp_inter_trial_delay: float = 0.0
    timestamp_inter_sequence_delay: float = 0.0

class pBART:
    def __init__(self):
        # username input
        self.subject_id = ""
        self.username_input = ""
        self.cursor_visible = True
        self.cursor_timer = 0
        self.session_trial_count = 0 # trial counter for the entire session
        
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Pygame Primate Balloon Analogue Risk Task (pBART)")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 72)
        self.font_medium = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # INITIALIZE Trial() dataclass
        self.trial = Trial()
        # initailize trial_number <- total trials in session
        self.trial.trial_number = self.session_trial_count = 0 
        self.game_state = GameState.WELCOME
        self.timer = 0
        self.post_reveal_delay = 60  # frames (200ms at 60 FPS)
        self.inter_trial_delay = 60  # frames (1000ms at 60 FPS)
        self.inter_sequence_delay = 60  # 1 second at 60 FPS
        
        # NEW: Track choice onset time for reaction time
        self.choice_onset_frame = 0
        
        # UI dimensions
        self.annulus_center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
        self.annulus_outer_radius = 150
        self.annulus_inner_radius = 100
        self.option_radius = 60
        self.left_option_pos = (SCREEN_WIDTH // 4, SCREEN_HEIGHT // 2)
        self.right_option_pos = (3 * SCREEN_WIDTH // 4, SCREEN_HEIGHT // 2)
        
        # Blackjack-style token distribution weights
        # TODO: RANDOMIZE token_weights on shuffle
        # TODO: RANDOMIZE LENGTH OF HANDS BETWEEN SHUFFLES (6-12 HANDS PER BLOCK)
        # 	HIGH RISK DECK, self.token_weights = [1]*9 + [4]
        # 	MEDIUM RISK DECK, token_weights = [1]*4 + [4] + [1]*5
        # 	LOW RISK DECK, token_weights = [4] + [1]*9
        # 	UNIFORM DECK? token_weights = [1]*10
        self.token_weights = [1]*9 + [4] # HIGH RISK DECK
        
        # NEW: Shuffle tracking
        self.hands_in_current_block = 0
        self.hands_until_shuffle = random.randint(6, 12)
        self.hands_until_shuffle = 5 #DEBUGGING
        self.deck_type = "HIGH"  # or "MEDIUM", "LOW", "UNIFORM"
        
        # Data tracking
        self.sequence_number = 0
        self.total_accumulated_tokens = 0
        self.trial_history = []
        
        self.data_dir = "data"
        self.ensure_data_directory()
        
        # NEW: Database setup
        self.db_connection = None
        self.db_cursor = None
        self.session_start_time = None
        self.initialize_database()
        
        # Randomize option positions each sequence
        self.randomize_option_positions()
        
        
    def randomize_deck(self):
        """Randomly select deck type and randomize hand count"""
        deck_options = {
            "HIGH": [1]*9 + [4],
            "MEDIUM": [1]*4 + [4] + [1]*5,
            "LOW": [4] + [1]*9,
            "UNIFORM": [1]*10
        }
        
        self.deck_type = random.choice(list(deck_options.keys()))
        self.token_weights = deck_options[self.deck_type]
        self.hands_until_shuffle = random.randint(6, 12)
        self.hands_until_shuffle = 5 # DEBUG
        
        
    def randomize_option_positions(self):
        """Randomly assign HIT and STAY to left or right"""
        if random.choice([True, False]):
            # HIT on left, STAY on right
            self.hit_pos = self.left_option_pos
            self.stay_pos = self.right_option_pos
            self.hit_key = "LEFT"
            self.stay_key = "RIGHT"
        else:
            # HIT on right, STAY on left
            self.hit_pos = self.right_option_pos
            self.stay_pos = self.left_option_pos
            self.hit_key = "RIGHT"
            self.stay_key = "LEFT"
        
    def ensure_data_directory(self):
        """Ensure data directory exists"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def initialize_database(self):
        """Initialize SQLite database with required tables"""
        db_filename = os.path.join(self.data_dir, "pbart_data.db")
        self.db_connection = sqlite3.connect(db_filename)
        self.db_cursor = self.db_connection.cursor()
        
        # Create sessions table
        self.db_cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_id TEXT NOT NULL,
                session_timestamp TEXT NOT NULL,
                total_sequences INTEGER,
                total_tokens_accumulated INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create trials table with GameState timestamps
        self.db_cursor.execute('''
            CREATE TABLE IF NOT EXISTS trials (
                trial_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                sequence_number INTEGER,
                trial_number INTEGER,
                choice TEXT,
                tokens_this_hit INTEGER,
                sequence_total INTEGER,
                result TEXT,
                total_accumulated INTEGER,
                reaction_time_ms REAL,
                trial_timestamp TEXT,
                timestamp_waiting_for_choice REAL,
                timestamp_revealing_outcome REAL,
                timestamp_post_reveal_delay REAL,
                timestamp_updating_annulus REAL,
                timestamp_bust REAL,
                timestamp_win REAL,
                timestamp_paused REAL,
                timestamp_inter_trial_delay REAL,
                timestamp_inter_sequence_delay REAL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        ''')
        
        self.db_connection.commit()

    def get_elapsed_time_ms(self):
        """Get elapsed time in milliseconds since session start"""
        if self.session_start_time is None:
            return 0.0
        return (pygame.time.get_ticks() - self.session_start_time)


    def record_gamestate_timestamp(self, gamestate: GameState):
        """Record the timestamp when a GameState is first entered"""
        elapsed_ms = self.get_elapsed_time_ms()
        
        if gamestate == GameState.WAITING_FOR_CHOICE:
            self.trial.timestamp_waiting_for_choice = elapsed_ms           
        elif gamestate == GameState.REVEALING_OUTCOME:
            self.trial.timestamp_revealing_outcome = elapsed_ms
        elif gamestate == GameState.POST_REVEAL_DELAY:
            self.trial.timestamp_post_reveal_delay = elapsed_ms
        elif gamestate == GameState.UPDATING_ANNULUS:
            self.trial.timestamp_updating_annulus = elapsed_ms
        elif gamestate == GameState.BUST:
            self.trial.timestamp_bust = elapsed_ms
        elif gamestate == GameState.WIN:
            self.trial.timestamp_win = elapsed_ms
        elif gamestate == GameState.PAUSED:
            self.trial.timestamp_paused = elapsed_ms
        elif gamestate == GameState.INTER_TRIAL_DELAY:
            self.trial.timestamp_inter_trial_delay = elapsed_ms
        elif gamestate == GameState.INTER_SEQUENCE_DELAY:
            self.trial.timestamp_inter_sequence_delay = elapsed_ms

    """"SQLite database"""
    def log_trial(self, trial: Trial):
        """"Log trial data to SQLite database"""
        trial.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # ADD THIS
        self.db_cursor.execute('''
            INSERT INTO trials (
                session_id, sequence_number, trial_number, choice, tokens_this_hit,
                sequence_total, result, total_accumulated, reaction_time_ms, trial_timestamp,
                timestamp_waiting_for_choice, timestamp_revealing_outcome, timestamp_post_reveal_delay,
                timestamp_updating_annulus, timestamp_bust, timestamp_win, timestamp_paused,
                timestamp_inter_trial_delay, timestamp_inter_sequence_delay
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            self.session_id,
            trial.sequence_number,
            trial.trial_number,
            trial.choice,
            trial.tokens_this_hit,
            trial.earned_tokens,
            trial.result,
            trial.total_accumulated,
            trial.reaction_time_ms,
            trial.timestamp,
            trial.timestamp_waiting_for_choice,
            trial.timestamp_revealing_outcome,
            trial.timestamp_post_reveal_delay,
            trial.timestamp_updating_annulus,
            trial.timestamp_bust,
            trial.timestamp_win,
            trial.timestamp_paused,
            trial.timestamp_inter_trial_delay,
            trial.timestamp_inter_sequence_delay
        ))
        self.db_connection.commit()


    def save_session_data(self):
        """Update session metadata in SQLite database"""
        # Only update if a session was actually created
        if not hasattr(self, 'session_id'):
            print("No session to save (user quit before entering username)")
            return
        
        self.db_cursor.execute('''
            UPDATE sessions 
            SET total_sequences = ?, total_tokens_accumulated = ?
            WHERE session_id = ?
        ''', (
            self.sequence_number,
            self.total_accumulated_tokens,
            self.session_id
        ))
        self.db_connection.commit()
        print(f"Session data updated for subject {self.subject_id}")


    def close_database(self):
        """Close the database connection"""
        if self.db_connection:
            self.db_connection.close()
        
        
    def draw_welcome_screen(self):
        """Draw the welcome/instructions screen"""
        self.screen.fill(GRAY_BG)
        
        # Title
        title = self.font_large.render("Balloon Analogue Risk Task", True, BLACK)
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 80))
        self.screen.blit(title, title_rect)
        
        # Instructions
        instructions = [
            "GOAL: Accumulate as many points as possible",
            "",
            "RULES:",
            "• Each hand starts at 0 points",
            "• Use the left and right arrows to select,",
            "• Press HIT (filled circle): reveal 1-10 points (risky!)",
            "• Press STAY (empty circle): cash out your points",
            "",
            "OUTCOMES:",
            "• If you choose STAY with 20 points or less → WIN!",
            "• If you choose HIT and exceed 20 points → BUST (lose all!)",
            "• Press STAY anytime to cash out safely",
            "",
            "NOTE:",
            "• The deck is shuffled every 6-12 hands."
            "",
            "Press SPACE to begin"
        ]
        
        y_position = 180
        for line in instructions:
            if line == "":
                y_position += 20
                continue
            
            # Bold key lines
            if line.startswith("GOAL:") or line.startswith("RULES:") or line.startswith("OUTCOMES:"):
                text = self.font_medium.render(line, True, BLACK)
            else:
                text = self.font_small.render(line, True, BLACK)
            
            text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, y_position))
            self.screen.blit(text, text_rect)
            y_position += 35
        
        pygame.display.flip()
        
        
    def draw_username_screen(self):
        """Draw username input screen"""
        self.screen.fill(GRAY_BG)
        
        # Title
        title = self.font_large.render("Enter Your Username", True, BLACK)
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 150))
        self.screen.blit(title, title_rect)
        
        # Instructions
        instruction = self.font_small.render("Type your name and press ENTER", True, BLACK)
        instruction_rect = instruction.get_rect(center=(SCREEN_WIDTH // 2, 250))
        self.screen.blit(instruction, instruction_rect)
        
        # Input box
        input_box_rect = pygame.Rect(SCREEN_WIDTH // 2 - 200, 350, 400, 60)
        pygame.draw.rect(self.screen, WHITE, input_box_rect)
        pygame.draw.rect(self.screen, BLACK, input_box_rect, 3)
        
        # Display username text
        username_text = self.font_medium.render(self.username_input, True, BLACK)
        username_rect = username_text.get_rect(center=(input_box_rect.centerx, input_box_rect.centery))
        self.screen.blit(username_text, username_rect)
        
        # Blinking cursor
        if self.cursor_visible:
            cursor_x = input_box_rect.x + 20 + username_text.get_width()
            pygame.draw.line(self.screen, BLACK, (cursor_x, input_box_rect.y + 10), 
                            (cursor_x, input_box_rect.y + 50), 2)
        
        # Hint text
        hint = self.font_small.render("(Backspace to delete, ENTER to continue)", True, BLACK)
        hint_rect = hint.get_rect(center=(SCREEN_WIDTH // 2, 450))
        self.screen.blit(hint, hint_rect)
        
        pygame.display.flip()
    
    
    def draw_annulus(self, surface, center, outer_radius, inner_radius, fill_percentage, fill_color=BLACK, show_bust=False, bust_amount=0):
        """
        Draw a filled annulus (ring) with a filled portion starting from the right (0 degrees)
        On BUST: show full black ring with red portion extending past (proportional to overage)
        """
        # Draw unfilled annulus outline first
        pygame.draw.circle(surface, BLACK, center, outer_radius, 3)
        pygame.draw.circle(surface, BLACK, center, inner_radius, 3)
        
        if show_bust and bust_amount > 0:
            # BUST display: full black ring + red portion extending past
            # First draw full black annulus (filled to 100%)
            angle_to_draw = 360
            
            points = [center]
            steps = max(int(angle_to_draw * 2), 1)
            
            for i in range(steps + 1):
                current_angle = (i / steps) * math.radians(angle_to_draw) - math.pi / 2
                x = center[0] + outer_radius * math.cos(current_angle)
                y = center[1] + outer_radius * math.sin(current_angle)
                points.append((x, y))
            
            for i in range(steps, -1, -1):
                current_angle = (i / steps) * math.radians(angle_to_draw) - math.pi / 2
                x = center[0] + inner_radius * math.cos(current_angle)
                y = center[1] + inner_radius * math.sin(current_angle)
                points.append((x, y))
            
            if len(points) > 2:
                pygame.draw.polygon(surface, BLACK, points)
            
            # Now draw red portion extending past the full circle
            bust_percentage = min(bust_amount / 20.0, 1.0)
            red_angle = 360 * bust_percentage
            
            points = [center]
            steps = max(int(red_angle * 2), 1)
            
            for i in range(steps + 1):
                current_angle = (i / steps) * math.radians(red_angle) - math.pi / 2
                x = center[0] + outer_radius * math.cos(current_angle)
                y = center[1] + outer_radius * math.sin(current_angle)
                points.append((x, y))
            
            for i in range(steps, -1, -1):
                current_angle = (i / steps) * math.radians(red_angle) - math.pi / 2
                x = center[0] + inner_radius * math.cos(current_angle)
                y = center[1] + inner_radius * math.sin(current_angle)
                points.append((x, y))
            
            if len(points) > 2:
                pygame.draw.polygon(surface, RED, points)
        else:
            # Normal display: fill black portion up to current tokens
            if fill_percentage > 0:
                angle_to_draw = min(360 * fill_percentage, 360)
                
                points = [center]
                steps = max(int(angle_to_draw * 2), 1)
                
                for i in range(steps + 1):
                    current_angle = (i / steps) * math.radians(angle_to_draw) - math.pi / 2
                    x = center[0] + outer_radius * math.cos(current_angle)
                    y = center[1] + outer_radius * math.sin(current_angle)
                    points.append((x, y))
                
                for i in range(steps, -1, -1):
                    current_angle = (i / steps) * math.radians(angle_to_draw) - math.pi / 2
                    x = center[0] + inner_radius * math.cos(current_angle)
                    y = center[1] + inner_radius * math.sin(current_angle)
                    points.append((x, y))
                
                if len(points) > 2:
                    pygame.draw.polygon(surface, BLACK, points)
    
    def draw_circle_option(self, surface, pos, radius, filled=False, color=BLACK):
        """Draw a circle option (filled or unfilled)"""
        if filled:
            pygame.draw.circle(surface, color, pos, radius)
        else:
            pygame.draw.circle(surface, color, pos, radius, 3)
    
    def draw_game(self):
        """Render all game elements"""
        # Draw welcome or username screens
        if self.game_state == GameState.WELCOME:
            self.draw_welcome_screen()
            return
        
        if self.game_state == GameState.USERNAME_INPUT:
            self.cursor_timer += 1
            if self.cursor_timer % 30 == 0:
                self.cursor_visible = not self.cursor_visible
            self.draw_username_screen()
            return
        
        if self.game_state == GameState.INTER_SEQUENCE_DELAY:
            self.screen.fill(GRAY_BG)
            pygame.display.flip()
            return
        
        self.screen.fill(GRAY_BG)
        
        # Draw annulus (token progress bar) - always draw it
        if self.game_state == GameState.BUST:
            bust_amount = self.trial.earned_tokens - 20
            self.draw_annulus(
                self.screen,
                self.annulus_center,
                self.annulus_outer_radius,
                self.annulus_inner_radius,
                1.0,
                BLACK,
                show_bust=True,
                bust_amount=bust_amount
            )
        else:
            fill_percentage = min(self.trial.earned_tokens / 20.0, 1.0)
            self.draw_annulus(
                self.screen,
                self.annulus_center,
                self.annulus_outer_radius,
                self.annulus_inner_radius,
                fill_percentage,
                BLACK
            )
        
        # Draw earned tokens text below the annulus
        earned_text = self.font_large.render(f"{self.trial.earned_tokens}", True, BLACK)
        earned_rect = earned_text.get_rect(center=(self.annulus_center[0], self.annulus_center[1] + 220))
        self.screen.blit(earned_text, earned_rect)
        
        # Draw total accumulated tokens (top right)
        total_text = self.font_medium.render(f"Total: {self.total_accumulated_tokens}", True, BLACK)
        total_rect = total_text.get_rect(topright=(SCREEN_WIDTH - 20, 20))
        self.screen.blit(total_text, total_rect)
        
        # Draw sequence counter (top left)
        sequences_text = self.font_small.render(f"Sequences: {self.sequence_number}", True, BLACK)
        sequences_rect = sequences_text.get_rect(topleft=(20, 20))
        self.screen.blit(sequences_text, sequences_rect)
        
        # Draw trial counter (top left)
        sequences_text = self.font_small.render(f"Trial #: {[self.session_trial_count, self.trial.trial_number]}", True, BLACK)
        sequences_rect = sequences_text.get_rect(topleft=(20, 40))
        self.screen.blit(sequences_text, sequences_rect)
        
        # Draw options based on game state
        if self.game_state == GameState.WAITING_FOR_CHOICE:
            # Draw HIT option (filled black circle)
            self.draw_circle_option(self.screen, self.hit_pos, self.option_radius, filled=True, color=BLACK)
            hit_label = self.font_small.render("HIT", True, WHITE)
            hit_label_rect = hit_label.get_rect(center=(self.hit_pos[0], self.hit_pos[1] + 100))
            self.screen.blit(hit_label, hit_label_rect)
            
            # Draw STAY option (unfilled circle)
            self.draw_circle_option(self.screen, self.stay_pos, self.option_radius, filled=False, color=BLACK)
            stay_label = self.font_small.render("STAY", True, BLACK)
            stay_label_rect = stay_label.get_rect(center=(self.stay_pos[0], self.stay_pos[1] + 100))
            self.screen.blit(stay_label, stay_label_rect)
            
            # Instructions - draw legend style with actual circles
            instruction_y = 50
            instruction_x_start = SCREEN_WIDTH // 2 - 150

            # Draw filled circle for HIT
            pygame.draw.circle(self.screen, BLACK, (instruction_x_start, instruction_y), 8)
            hit_text = self.font_small.render("HIT", True, BLACK)
            self.screen.blit(hit_text, (instruction_x_start + 20, instruction_y - 16))

            # Draw open circle for STAY
            pygame.draw.circle(self.screen, BLACK, (instruction_x_start + 120, instruction_y), 8, 2)
            stay_text = self.font_small.render("STAY", True, BLACK)
            self.screen.blit(stay_text, (instruction_x_start + 140, instruction_y - 16))
        
        elif self.game_state == GameState.REVEALING_OUTCOME:
            # Display the revealed token color and amount - NO HIT/STAY CIRCLES
            if self.trial.tokens_this_hit > 0:  # Safety check
                color = TOKEN_COLORS[self.trial.tokens_this_hit - 1]
                self.draw_circle_option(
                    self.screen,
                    self.hit_pos,
                    self.option_radius * 1.2,
                    filled=True,
                    color=color
                )
                
                outcome_text = self.font_medium.render(f"+{self.trial.tokens_this_hit} tokens", True, BLACK)
                outcome_rect = outcome_text.get_rect(center=(self.hit_pos[0], self.hit_pos[1] + 100))
                self.screen.blit(outcome_text, outcome_rect)

        elif self.game_state == GameState.POST_REVEAL_DELAY:
            # Show the revealed card still - NO HIT/STAY CIRCLES
            if self.trial.tokens_this_hit > 0:  # Safety check
                color = TOKEN_COLORS[self.trial.tokens_this_hit - 1]
                self.draw_circle_option(
                    self.screen,
                    self.hit_pos,
                    self.option_radius * 1.2,
                    filled=True,
                    color=color
                )
                
                outcome_text = self.font_medium.render(f"+{self.trial.tokens_this_hit} tokens", True, BLACK)
                outcome_rect = outcome_text.get_rect(center=(self.hit_pos[0], self.hit_pos[1] + 100))
                self.screen.blit(outcome_text, outcome_rect)
        
        elif self.game_state == GameState.BUST:
            # Display BUST message
            bust_text = self.font_large.render("BUST!", True, RED)
            bust_rect = bust_text.get_rect(center=(SCREEN_WIDTH // 2, 50))
            self.screen.blit(bust_text, bust_rect)
            
            loss_text = self.font_medium.render("All tokens lost", True, RED)
            loss_rect = loss_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 150))
            self.screen.blit(loss_text, loss_rect)
            
            restart_text = self.font_small.render("Press SPACE to start new sequence", True, BLACK)
            restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 80))
            self.screen.blit(restart_text, restart_rect)
        
        elif self.game_state == GameState.WIN:
            # Display WIN message
            win_text = self.font_large.render("WIN!", True, GREEN)
            win_rect = win_text.get_rect(center=(SCREEN_WIDTH // 2, 50))
            self.screen.blit(win_text, win_rect)
            
            reward_text = self.font_medium.render(f"Earned {self.trial.earned_tokens} tokens", True, GREEN)
            reward_rect = reward_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 150))
            self.screen.blit(reward_text, reward_rect)
            
            restart_text = self.font_small.render("Press SPACE to start new sequence", True, BLACK)
            restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 80))
            self.screen.blit(restart_text, restart_rect)
        elif self.game_state == GameState.PAUSED:
            self.screen.fill(WHITE)
            
            shuffle_text = self.font_large.render("SHUFFLING", True, BLACK)
            shuffle_rect = shuffle_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50))
            self.screen.blit(shuffle_text, shuffle_rect)

        pygame.display.flip()
    
    def handle_hit(self):
        """Handle HIT option chosen"""
        self.trial.choice = "HIT"
        self.trial.tokens_this_hit = random.choices(range(1, 11), weights=self.token_weights, k=1)[0]
        
        # Calculate reaction time
        frames_elapsed = self.timer - self.choice_onset_frame
        self.trial.reaction_time_ms = (frames_elapsed / 60.0) * 1000
        
        self.game_state = GameState.REVEALING_OUTCOME
        self.record_gamestate_timestamp(GameState.REVEALING_OUTCOME)
        self.timer = 0
    
    def handle_stay(self):
        """Handle STAY option chosen - player keeps earned tokens"""
        self.trial.choice = "STAY"
        self.trial.result = "WIN"
        self.trial.tokens_this_hit = 0
        
        # Calculate reaction time
        frames_elapsed = self.timer - self.choice_onset_frame
        self.trial.reaction_time_ms = (frames_elapsed / 60.0) * 1000
        
        self.trial.trial_number += 1  # ADD THIS - increment before WIN
        
        self.hands_in_current_block += 1  # ADD THIS - increment before WIN
        
        # Increment hand counter for WIN too!
        self.total_accumulated_tokens += self.trial.earned_tokens
        self.trial.total_accumulated = self.total_accumulated_tokens
        
        # Record annulus update, then WIN
        self.record_gamestate_timestamp(GameState.UPDATING_ANNULUS)
        self.game_state = GameState.WIN
        self.record_gamestate_timestamp(GameState.WIN)
        self.timer = 0
        
    def reset_sequence(self):
        """Reset for a new sequence"""
        self.sequence_number += 1
        self.randomize_option_positions()
        self.trial = Trial()
        self.trial.trial_number = 0 # do not reset
        self.trial.sequence_number = self.sequence_number
        
        # Explicitly ensure all timestamps are 0
        self.trial.timestamp_waiting_for_choice = 0.0
        self.trial.timestamp_revealing_outcome = 0.0
        self.trial.timestamp_post_reveal_delay = 0.0
        self.trial.timestamp_updating_annulus = 0.0
        self.trial.timestamp_bust = 0.0
        self.trial.timestamp_win = 0.0
        self.trial.timestamp_paused = 0.0
        self.trial.timestamp_inter_trial_delay = 0.0
        self.trial.timestamp_inter_sequence_delay = 0.0
        
        self.game_state = GameState.WAITING_FOR_CHOICE
        self.record_gamestate_timestamp(GameState.WAITING_FOR_CHOICE)
        print("")
        print(f"Starting trial: {self.session_trial_count}")
        
        self.timer = 0
        self.choice_onset_frame = 0
    
    def handle_input(self):
        """Handle keyboard and quit events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                # print(f"key down")
                # Username input handling
                # SPACE to progress from WELCOME to USERNAME_INPUT (CHECK FIRST!)
                if event.key == pygame.K_SPACE and self.game_state == GameState.WELCOME:
                    self.game_state = GameState.USERNAME_INPUT
                    return True
                elif self.game_state == GameState.USERNAME_INPUT:
                    if event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                        if len(self.username_input) > 0:
                            self.subject_id = self.username_input
                            
                            # NEW: Create session NOW that we have subject_id
                            if self.session_start_time is None:
                                self.session_start_time = pygame.time.get_ticks()
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                self.db_cursor.execute('''
                                    INSERT INTO sessions (subject_id, session_timestamp, total_sequences, total_tokens_accumulated)
                                    VALUES (?, ?, ?, ?)
                                ''', (self.subject_id, timestamp, 0, 0))
                                self.db_connection.commit()
                                self.session_id = self.db_cursor.lastrowid
                            
                            self.reset_sequence()
                            return True
                        else:
                            print(f"DEBUG: Username is empty, ignoring RETURN")
                    elif event.key == pygame.K_BACKSPACE:
                        self.username_input = self.username_input[:-1]
                    # ADD THIS BLOCK:
                    elif event.unicode.isalnum() or event.unicode == ' ':
                        if len(self.username_input) < 20:
                            self.username_input += event.unicode
                # Rest of input handling...
                elif event.key == pygame.K_LEFT and self.game_state == GameState.WAITING_FOR_CHOICE:
                    if self.hit_pos == self.left_option_pos:
                        self.handle_hit()
                    else:
                        self.handle_stay()

                elif event.key == pygame.K_RIGHT and self.game_state == GameState.WAITING_FOR_CHOICE:
                    if self.hit_pos == self.right_option_pos:
                        self.handle_hit()
                    else:
                        self.handle_stay()
                
                elif event.key == pygame.K_SPACE and self.game_state in [GameState.WIN, GameState.BUST]:
                    self.game_state = GameState.INTER_SEQUENCE_DELAY
                    self.record_gamestate_timestamp(GameState.INTER_SEQUENCE_DELAY)  # Record RIGHT HERE 
                    self.log_trial(self.trial)  # ADD THIS - log AFTER recording timestamp
                    self.timer = 0
                
                elif event.key == pygame.K_ESCAPE:
                    return False
        
        return True


    def update(self):
        """Update game logic with timing"""
        self.timer += 1
        
        if self.game_state == GameState.REVEALING_OUTCOME:
            # Wait for delay frames before moving to POST_REVEAL_DELAY
            if self.timer >= self.post_reveal_delay:
                #print(f"DEBUG UPDATE: Transitioning to POST_REVEAL_DELAY")
                self.game_state = GameState.POST_REVEAL_DELAY
                self.record_gamestate_timestamp(GameState.POST_REVEAL_DELAY)  # Record RIGHT HERE 
                self.timer = 0
        
        elif self.game_state == GameState.POST_REVEAL_DELAY:
            if self.timer >= 1:
                self.trial.earned_tokens += self.trial.tokens_this_hit
                
                # Record when annulus starts updating
                self.record_gamestate_timestamp(GameState.UPDATING_ANNULUS)
                self.timer = 0
                
                if self.trial.earned_tokens > 20:
                    self.trial.result = "BUST"
                    self.trial.total_accumulated = self.total_accumulated_tokens
                    self.trial.trial_number += 1  # ADD THIS
                    self.hands_in_current_block += 1  # ADD THIS
                    if self.hands_in_current_block >= self.hands_until_shuffle:
                        self.game_state = GameState.PAUSED
                        self.record_gamestate_timestamp(GameState.PAUSED)
                        self.log_trial(self.trial)  # ADD THIS
                    else:
                        self.game_state = GameState.BUST
                        self.record_gamestate_timestamp(GameState.BUST)
                    
                    self.timer = 0
                    
                else:  # CONTINUE
                    self.trial.result = "CONTINUE"
                    self.trial.total_accumulated = self.total_accumulated_tokens
                    
                    # Increment hand counter
                    # Check if shuffle is needed BEFORE creating new trial
                    if self.hands_in_current_block >= self.hands_until_shuffle:
                        print(f"DEBUG: SHUFFLE TRIGGERED!")
                        # Time to shuffle!
                        self.game_state = GameState.PAUSED
                        self.record_gamestate_timestamp(GameState.PAUSED)
                        self.log_trial(self.trial)  # Log AFTER recording state
                        self.timer = 0
                    else:
                        self.game_state = GameState.INTER_TRIAL_DELAY
                        self.record_gamestate_timestamp(GameState.INTER_TRIAL_DELAY)
                        self.log_trial(self.trial)  # Log AFTER recording state
                        print(f"End of trial: {[self.session_trial_count,self.trial.trial_number]}")
                        print("")
                        earned_tokens_carry = self.trial.earned_tokens

                        self.trial = Trial()
                        self.trial.sequence_number = self.sequence_number
                        self.trial.earned_tokens = earned_tokens_carry
                        self.session_trial_count += 1
                        self.trial.trial_number = self.session_trial_count
                        self.timer = 0
        
        
        elif self.game_state == GameState.INTER_TRIAL_DELAY: 
            # Wait before next trial
            if self.timer >= self.inter_trial_delay:
                self.randomize_option_positions()
                # Increment trial number for next trial
                self.trial.trial_number += 1
                self.hands_in_current_block += 1
                
        
                # Continue to next trial normally
                self.game_state = GameState.WAITING_FOR_CHOICE
                self.record_gamestate_timestamp(GameState.WAITING_FOR_CHOICE)
                self.choice_onset_frame = 0
                self.timer = 0
                print(f"Start trial: {self.session_trial_count}")
           
        elif self.game_state == GameState.INTER_SEQUENCE_DELAY:
            # only on [GameState.WIN, GameState.BUST]
            # Wait 1 second before starting next sequence
            self.timer += 1
            if self.timer >= self.inter_sequence_delay:
                self.reset_sequence()
                self.timer = 0
                print(f"End trial: {self.session_trial_count}")
        elif self.game_state == GameState.PAUSED:
            if self.timer >= 120:
                self.hands_in_current_block = 0
                print(f"DEBUG: trial {self.trial.trial_number} of {self.hands_in_current_block}")
                self.randomize_deck()
                self.game_state = GameState.INTER_SEQUENCE_DELAY
                self.record_gamestate_timestamp(GameState.INTER_SEQUENCE_DELAY)
                self.timer = 0
            else:
                self.timer += 1
                
    def quit_game(self):
        """Clean exit"""
        self.save_session_data()
        self.close_database()
        pygame.quit()
    
    def run(self):
        """Main game loop"""
        running = True
        self.game_state = GameState.WELCOME
        frame_count = 0
        
        # DON'T create session here - wait until username is entered
        
        try:
            while running:
                frame_count += 1         
                running = self.handle_input()
                self.update()
                self.draw_game()
                self.clock.tick(FPS)
                
        finally:
            self.save_session_data()
            self.close_database()
            pygame.quit()

if __name__ == "__main__":
    game = pBART()
    game.run()
    
#     # Ask if user wants to run analysis
#     response = input("\nRun analysis? (y/n): ")
#     if response.lower() == 'y':
#         from analyze_pbart import pBARTAnalyzer
#         analyzer = pBARTAnalyzer()
#         analyzer.run_analysis()
