import re 
import polars as pl

# define a function that inspects the speech text and returns the event type
# we run this on each chunk before it becomes a row
def get_event(speech):
    if re.search(r'\*1\*', speech):
        # even number of *1* markers = self-contained, odd = still open
        complete = len(re.findall(r'\*1\*', speech)) % 2 == 0
        return "Reading Story", complete
    if re.search(r'\*2\*', speech):
        complete = len(re.findall(r'\*2\*', speech)) % 2 == 0
        return "Reading Poem", complete
    if re.search(r'[<>]', speech):
        # complete if both < and > are present in the same chunk
        complete = bool(re.search(r'<', speech) and re.search(r'>', speech))
        return "Citing Text", complete
    if re.search(r'"', speech):
        # even number of quotes = self-contained
        complete = len(re.findall(r'"', speech)) % 2 == 0
        return "Citing GM", complete
    return None, False


# read the whole file as one string

def parse_transcript(filepath):
    text = open(filepath).read()

    # split on the pattern "Name: " at the start of a line
    # each chunk contains one speaker's full turn, possibly across multiple lines
    chunks = re.split(r"\n(?=\w[\w\s]*:)", text)
    rows = []
    current_event = None # define switch for long events
    for chunk in chunks:
        match = re.match(r"^(\w[\w\s]*):\s+(.+?)(?:\s+(\d{2}:\d{2}:\d{2}[\.\d]*))?$", chunk, re.DOTALL)
        if match:
            speaker, speech, time = match.groups()
            event, is_complete = get_event(speech)

            if event is not None:
                if current_event == event:
                    # we're already inside this block — this is the closing marker
                    current_event = None
                elif not is_complete:
                    # opening marker without a close — start tracking
                    current_event = event
                # if is_complete, it's self-contained — just assign it, don't toggle
            else:
                # no marker — inherit ongoing event if inside a block
                event = current_event

            rows.append((speaker, speech, time, event))

    return pl.DataFrame(
        rows,
        schema=["speaker", "speech", "time", "event"],
        orient="row"
    )

def parse_transcript_xlsx(filepath):
    # read_excel loads the xlsx into a polars df
    # it infers column names from the first row, so we rename them
    df = pl.read_excel(filepath, has_header=False)
    df = df.with_columns(pl.all().cast(pl.String))
    if df["column_1"][0] == "1":
        df = parse_transcript_maxqda_xlsx(filepath)
        return df
    # this is special for transcript nr 2 - because there were some issues with the transcription in excel and now the timestamps need to be reformatted 
    if len(df.columns) > 2: 
        #df = df.with_columns(pl.all().cast(pl.String))  # cast to string before comparing
        df = df.with_columns(
                pl.nth(2)
                    .str.replace(r"(?:\d+:)?1899-12-31 ", "")   # remove the date part
                    .str.replace(r"(\.\d)0+$", "$1")             # trim trailing zeros: .500 → .5
                    .alias(df.columns[2])
        )
        df = df.drop(pl.nth(3))
        df = df.rename({df.columns[0]: "speaker", df.columns[1]: "speech", df.columns[2]: "time"})

    else:
        # rename whatever the columns are called to something predictable
        df = df.rename({df.columns[0]: "speaker", df.columns[1]: "raw"})
        
        # drop empty rows
        df = df.filter(pl.col("raw").str.strip_chars().str.len_chars() > 0)
        
        # extract timestamp from the end of the raw column
        df = df.with_columns([
            pl.col("raw")
            .str.extract(r"(\d{2}:\d{2}:\d{2}[\.\d]*)$", group_index=1)
            .alias("time")
        ])
        df = df.with_columns([
            pl.when(pl.col("time").is_not_null())
            .then(
                pl.col("raw")
                    .str.extract(r"(.+?)\s+\d{2}:\d{2}:\d{2}[\.\d]*$", group_index=1)
                    .str.strip_chars()
            )
            .otherwise(pl.col("raw").str.strip_chars())  # no timestamp — use the whole raw text
            .alias("speech")
        ])
    
    # forward fill speaker in case of empty cells between turns
    df = df.with_columns(pl.col("speaker").forward_fill())
    
    # apply the same event detection as the txt parser
    current_event = None
    rows = []
    for row in df.iter_rows(named=True):  # named=True lets you access by column name
        event, is_complete = get_event(row["speech"]) if row["speech"] else (None, False)
        
        if event is not None:
            if current_event == event:
                current_event = None
            elif not is_complete:
                current_event = event
        else:
            event = current_event
        
        rows.append((row["speaker"], row["speech"], row["time"], event))
    
    return pl.DataFrame(
        rows,
        schema=["speaker", "speech", "time", "event"],
        orient="row"
    )

def check_event_markers(transcripts):
    """
    Checks all transcripts for unclosed event markers.
    Returns a summary of issues found.
    """
    issues = []

    for session, df in transcripts.items():
        
        # count total markers across the whole session
        full_text = " ".join(df["speech"].drop_nulls().to_list())
        
        counts = {
            "*1*": len(re.findall(r'\*1\*', full_text)),
            "*2*": len(re.findall(r'\*2\*', full_text)),
            '"':   len(re.findall(r'"', full_text)),
            "<":   len(re.findall(r'<', full_text)),
            ">":   len(re.findall(r'>', full_text)),
        }
        
        # check for imbalances
        if counts["*1*"] % 2 != 0:
            issues.append(f"{session}: odd number of *1* markers ({counts['*1*']})")
        if counts["*2*"] % 2 != 0:
            issues.append(f"{session}: odd number of *2* markers ({counts['*2*']})")
        quote = '"'
        if counts['"'] % 2 != 0:
            issues.append(f"{session}: odd number of {quote} markers ({counts[quote]})")
        if counts["<"] != counts[">"]:
            issues.append(f"{session}: {counts['<']} opening < but {counts['>']} closing >")

            if counts["<"] > counts[">"]:
                closed = True
                issue = False
                for row in df.iter_rows(named=True):
                    if not row["speech"]:
                        continue
                    for char in row["speech"]:
                        if (char == "<") and (closed == True):
                            closed = False
                        elif (char == ">") and (closed == False):
                            closed = True
                        elif (char == "<") and (closed == False):
                            issues.append(f"unclosed before this: '{row['speech']}'")
                                        
            if counts[">"] > counts["<"]:
                opened = False
                for row in df.iter_rows(named=True):
                    if not row["speech"]:
                        continue
                    for char in row["speech"]:
                        if (char == "<") and (opened == False):
                            opened = True
                        elif (char == ">") and (opened == True):
                            opened = False
                        elif (char == "<") and (opened == True):
                            issues.append(f"unopened before this: '{row['speech']}'")
                            



    if not issues:
        print("All markers are balanced across all transcripts!")
    else:
        print(f"Found {len(issues)} issue(s):\n")
        for issue in issues:
            print(issue)

def split_on_markers(speech):
    """
    Splits a speech string into segments, isolating *1* and *2* blocks.
    e.g. "blabla *1* text *1* blabla" → ["blabla", "*1* text *1*", "blabla"]
    """
    # re.split with a capturing group keeps the matched part in the result
    parts = re.split(r'(\*1\*.*?\*1\*|\*2\*.*?\*2\*)', speech, flags=re.DOTALL)
    # strip whitespace and remove empty strings
    return [p.strip() for p in parts if p.strip()]

def sync_speakers(df):
    df = (
        df
        .drop("event")
        # create a turn counter: compare each speaker to the previous row's speaker
        # when they differ, mark as 1 (new turn), otherwise 0
        # cumsum() then gives each consecutive run a unique number
        .with_columns(
            (pl.col("speaker") != pl.col("speaker").shift(1))
            .fill_null(True)   # first row has no previous, treat as a new turn
            .cum_sum()
            .alias("turn")
        )
        # now group by turn (which implicitly groups consecutive same-speaker rows)
        .group_by("turn", maintain_order=True)
        .agg(
            pl.col("speaker").first(),           # speaker is the same across the run
            pl.col("speech").str.join(" "),      # join all speech lines with a space
            pl.col("time").last()                # keep the last timestamp of the turn
        )
        .drop("turn")                            # no longer needed
    )
    # now split rows that contain *1* or *2* blocks
    rows = []
    for row in df.iter_rows(named=True):
        parts = split_on_markers(row["speech"])
        if len(parts) > 1:
            for part in parts:
                rows.append((row["speaker"], part, row["time"]))
        else:
            rows.append((row["speaker"], row["speech"], row["time"]))

    return pl.DataFrame(rows, schema=["speaker", "speech", "time"], orient="row")


def parse_transcript_maxqda_xlsx(filepath):
    df = pl.read_excel(filepath, has_header=False)
    
    # two columns: row numbers and content — we only need the second column
    lines = df[df.columns[1]].drop_nulls().to_list()
    text = "\n".join(str(l) for l in lines)
    
    chunks = re.split(r"\n(?=\[|\w[\w\s]*:)", text)
    
    current_event = None
    rows = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        
        match = re.match(
            r"^(?:\[(\d{1,2}:\d{2}:\d{2}[\.\d]*)\])?\s*(\w[\w\s]*):\s*(.+)$",
            chunk,
            re.DOTALL
        )
        if match:
            time, speaker, speech = match.groups()
            speech = speech.strip()
            
            event, is_complete = get_event(speech) if speech else (None, False)
            
            if event is not None:
                if current_event == event:
                    current_event = None
                elif not is_complete:
                    current_event = event
            else:
                event = current_event
            
            rows.append((speaker.strip(), speech, time, event))
    
    return pl.DataFrame(
    rows,
    schema={
        "speaker": pl.String,
        "speech": pl.String,
        "time": pl.String,
        "event": pl.String
    },
    orient="row"
    )