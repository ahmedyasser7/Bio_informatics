import streamlit as st
import pandas as pd
import numpy as np
from itertools import permutations

# Set the page configuration
st.set_page_config(page_title="Bio-Informatics", layout="wide", page_icon="🧬")

# Custom CSS to enhance the layout and colors
st.markdown("""
    <style>
        .css-1v0mbdj { background-color: #6918b4; }
        .css-1eg9s7v { background-color: #8600b3; }
        .css-1h6c3rr { background-color: #862d59; }
        .sidebar .sidebar-content { background-color: #862d59; }
        .css-1a2a0cz { font-size: 18px; color: #862d59; }
        .stTextInput>div>div>input { background-color: #6918b4; }
        .stTextArea>div>div>textarea { background-color: #6918b4; }
        .stButton>button { background-color: #6918b4; color: 6918b4; }
        .stSelectbox>div>div>input { background-color: #6918b4; }
        h1 { color: #6918b4; }
        .stMarkdown { color: #8600b3; }
        .stDataFrame { margin-bottom: 2em; }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Bio content")
page = st.sidebar.radio(
    "Select a page:", ["Home", "Data upload", "Apply on file", "Apply for pattern", "Contact Us"])

# Function for Home page


def home_page():
    st.title("Welcome to the BioInformatics App!")
    st.write(
        "This app is designed to help you achieve your Bio goals efficiently and effectively.")
    st.image("dna.jpg", use_column_width=True)

    st.markdown("""---""")


def apply_for_pattern():
    st.title("Try Our Algorithms ")
    algo = st.selectbox("choose what you want : ", [
                        "translation", "naive_match", "boyer_moore", "IndexSorted", "suffix_array", "overlap"])
    if algo == "translation":
        st.header("FASTA File/Sequence Translation")
        st.write("Upload a FASTA-like sequence, input the text manually, or enter pattern and text for processing.")

        fasta_input = st.text_area("Paste your FASTA-like text here:", height=200)
        process_type = st.selectbox("Choose Process Type:", [ "Pattern Match", "Sequence Extraction", "GC Content Calculation", "Table Export"])
        
        if process_type == "Pattern Match":
            pattern = st.text_input("Enter your pattern:")
            text = st.text_area("Enter the text to search in:")
            
            if pattern and text:
                st.subheader("Pattern Match Results:")
                match_indices = []
                for i in range(len(text) - len(pattern) + 1):
                    if text[i:i+len(pattern)] == pattern:
                        match_indices.append(i)
                
                if match_indices:
                    st.write(f"Pattern found at indices: {match_indices}")
                else:
                    st.write("No matches found.")
        elif fasta_input:
            tb = []  # For storing processed data
            lines = fasta_input.strip().split("\n")  # Split input into lines
            
            # Process input based on user choice
            if process_type == "Sequence Extraction":
                st.subheader("Extracted Sequences:")
                for i in range(0, len(lines), 2):
                    header = lines[i][1:]  # Remove ">"
                    sequence = lines[i + 1]
                    st.write(f"ID: {header}, Sequence: {sequence}")
                    tb.append([header, sequence])
                
                # Show a table
                st.write("Processed Table:")
                df = pd.DataFrame(tb, columns=["ID", "Sequence"])
                st.dataframe(df)
            
            elif process_type == "GC Content Calculation":
                def GC_Content(seq):
                    """Calculate GC content of a sequence."""
                    num_G = seq.count("G")
                    num_C = seq.count("C")
                    total_GC = num_G + num_C
                    return total_GC / len(seq) if len(seq) > 0 else 0
                
                st.subheader("GC Content Results:")
                for i in range(0, len(lines), 2):
                    header = lines[i][1:]  # Remove ">"
                    sequence = lines[i + 1]
                    gc_content = GC_Content(sequence)
                    st.write(f"ID: {header}, GC Content: {gc_content:.2%}")
                    tb.append([header, gc_content])
                
                # Show results in table format
                st.write("Processed GC Table:")
                df = pd.DataFrame(tb, columns=["ID", "GC_Content"])
                st.dataframe(df)
            
            elif process_type == "Table Export":
                st.subheader("Export Data as CSV")
                for i in range(0, len(lines), 2):
                    header = lines[i][1:]
                    sequence = lines[i + 1]
                    tb.append([header, sequence])
                
                df = pd.DataFrame(tb, columns=["ID", "Sequence"])
                st.write("Here is the data table:")
                st.dataframe(df)
                
                # Provide download button
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download CSV",
                    data=csv,
                    file_name="processed_fasta_data.csv",
                    mime="text/csv"
                )

                
    if algo == "naive_match":
        target = st.text_input("enter your pattern :")
        text = st.text_input("enter the text:")
        st.write("the index that occur match on is :")
        for i in range(len(text) - len(target) + 1):
            if text[i:i+len(target)] == target:
                st.write(i, " ")
                
    if algo == "suffix_array":

        text = st.text_input("enter the text:")
        dec = {
            '$': 0,
            'a': 1,
            'c': 2,
            'g': 3,
            't': 4
        }

        table = []
        i = 2**0
        n = 0
        while True:
            l = []
            dec2 = {}
            if i > 1:
                for j in range(len(text)):
                    if not (table[n-1][j:j+i] in l):
                        l.append(table[n-1][j:j+i])
                        l.sort()

                for j in range(len(l)):
                    dec2[tuple(l[j])] = j
            row = []
            for j in range(len(text)):
                if i == 1:
                    row.append(dec[text[j]])
                else:
                    row.append(dec2[tuple(table[n-1][j:j+i])])
            table.append(row)
            flag = 0
            for j in range(len(row)):
                c = 0
                c = row.count(j)
                if c > 1:
                    flag = 1
                    break

            if flag == 0:
                break
            n += 1
            i = 2**n
        st.write("suffix_table", row)

    if algo == "IndexSorted":
        text = st.text_input("enter your text :")
        target = st.number_input("enter the K value:", step=1)
        index = []
        for i in range(len(text)-target+1):
            index.append((text[i:i+target], i))
        index.sort()
        st.write(index)
    if algo == "boyer_moore":

        text = st.text_input("enter your text :")
        target = st.text_input("enter the pattern:")

        def boyer_moore(text, pattern):
            def preprocess_bad_character_rule(pattern):
                """Preprocess pattern for the bad character rule."""
                bad_char = {}
                for i in range(len(pattern)):
                    bad_char[pattern[i]] = i
                return bad_char

            def preprocess_good_suffix_rule(pattern):
                """Preprocess pattern for the good suffix rule."""
                m = len(pattern)
                good_suffix = [-1] * m
                z_suffix = [0] * m

            # Z algorithm to find suffix lengths
                left, right, z = 0, 0, [0] * m
                for i in range(1, m):
                    if i <= right:
                        z[i] = min(right - i + 1, z[i - left])
                    while i + z[i] < m and pattern[z[i]] == pattern[i + z[i]]:
                        z[i] += 1
                    if i + z[i] - 1 > right:
                        left, right = i, i + z[i] - 1

                for i in range(m - 1, -1, -1):
                    if i + z[i] == m:
                        for j in range(m - 1 - i):
                            if good_suffix[j] == -1:
                                good_suffix[j] = i
                for i in range(m - 1):
                    good_suffix[m - 1 - z[i]] = i

                return good_suffix

            n = len(text)
            m = len(pattern)
            bad_char = preprocess_bad_character_rule(pattern)
            good_suffix = preprocess_good_suffix_rule(pattern)

            matches = []
            shift = 0

            while shift <= n - m:
                j = m - 1

                # Compare from the end of the pattern
                while j >= 0 and pattern[j] == text[shift + j]:
                    j -= 1

                if j < 0:
                    matches.append(shift)
                    shift += m - good_suffix[0] if m > 1 else 1
                else:
                    bad_char_shift = j - bad_char.get(text[shift + j], -1)
                    good_suffix_shift = good_suffix[j] if j < m - 1 else 1
                    shift += max(bad_char_shift, good_suffix_shift)

            return matches

    # Example usage
        matches = boyer_moore(text, target)
        st.write("Pattern found at indices:", matches)

    if algo == "overlap":
        def overlap(a, b, min_length=3):
            start = 0
            while True:
                start = a.find(b[:min_length], start)
                if start == -1:
                    return 0
                if b.startswith(a[start:]):
                    return len(a) - start
                start += 1

        def native_overlap(reads, k):
            olap = {}
            for a, b in permutations(reads, 2):
                olen = overlap(a, b, k)
                if olen > 0:
                    olap[(a, b)] = olen
            return olap

        st.title("Sequence Overlap Finder")

        # Input for minimum overlap length
        st.header("Minimum Overlap Length")
        min_length = st.number_input(
            "Enter minimum overlap length (k):", min_value=1, value=3)

        # Dynamic sequence input
        st.header("Input Sequences")
        st.write(f"Enter at least {min_length} sequences:")
        sequences = []
        for i in range(min_length):
            seq = st.text_input(f"Sequence {i + 1}:", key=f"seq_{i}")
            if seq:
                sequences.append(seq)

        # Add additional sequence inputs
        additional_count = st.number_input(
            "Add more sequences (optional):", min_value=0, value=0, step=1
        )
        for j in range(additional_count):
            seq = st.text_input(f"Additional Sequence {j + 1}:", key=f"extra_seq_{j}")
            if seq:
                sequences.append(seq)

        # Process and display results
        if st.button("Find Overlaps"):
            if len(sequences) < min_length:
                st.error(f"Please enter at least {min_length} sequences.")
            else:
                overlaps = native_overlap(sequences, min_length)
                if overlaps:
                    st.subheader("Overlaps Found:")
                    for (a, b), olen in overlaps.items():
                        st.write(f"Overlap between {a} and {b}: {olen}")
                else:
                    st.write("No overlap found")


# Function for Data Upload page
def data_upload_page():
    st.title("Upload Data file")

    uploaded_file = st.file_uploader(
        "Upload your file (FASTA or TEXT):",
        type=["fasta", "txt"]
    )

    if uploaded_file:
        file_type = uploaded_file.name.split(".")[-1].lower()  # To get the file extension

        # FASTA Files
        if file_type in ["fasta"]:
            st.write(f"Here's the content of your {file_type.upper()} file:")
            fasta_content = {}
            current_seq = ""
            for line in uploaded_file:
                line = line.decode("utf-8").strip()
                if line.startswith(">"):
                    # Remove '>' from the sequence header
                    current_seq = line[1:]
                    fasta_content[current_seq] = ""
                else:
                    fasta_content[current_seq] += line
            cnt = 1
            for seq_name, sequence in fasta_content.items():
                st.text(f"Sequence {cnt} Name: {seq_name}")
                st.code(sequence)
                cnt += 1
            st.session_state[f"{file_type}_data"] = fasta_content

        # Text Files
        elif file_type == "txt":
            st.write("Here's the content of your text file:")

            text_content = uploaded_file.read().decode("utf-8")

            # Remove ">" and split into lines
            cleaned_content = text_content.replace(">", "")
            lines = cleaned_content.strip().split("\n")

            # Identify DNA sequences
            dna_sequences = [line.strip()
                            for line in lines if line.strip().isalpha()]

            # Display the cleaned content
            st.subheader("Cleaned Text Content:")
            st.text_area("Processed Text:", value="\n".join(lines), height=200)

            # Display DNA sequences
            st.subheader("Extracted DNA Sequences:")
            for idx, seq in enumerate(dna_sequences, start=1):
                st.text(f"Sequence {idx}:")
                st.code(seq)

            # Store in session state
            st.session_state["text_data"] = {
                "cleaned_text": cleaned_content, "dna_sequences": dna_sequences}

        else:
            st.error(
                "Unsupported file type! Please upload a CSV, FASTA, FNA, or TEXT file.")
    else:
        st.info("Please upload a file to proceed.")


def algorithm_apply():
    st.title("Apply an Algorithm for the uploaded file!")
    # Ensure data is available
    if "uploaded_data" not in st.session_state and \
        "fasta_data" not in st.session_state and \
        "text_data" not in st.session_state:
        st.warning("No data found! Please upload a file first.")
        return

    # Select File Type
    file_type = st.radio("Choose the type of uploaded data:", options=["FASTA", "Text"])

    selected_sequence = None
    if file_type == "FASTA" and "fasta_data" in st.session_state:
        fasta_data = st.session_state["fasta_data"]
        sequence_name = st.selectbox("Choose a sequence:", options=list(fasta_data.keys()))
        selected_sequence = fasta_data[sequence_name]
        st.text(f"Selected Sequence ({sequence_name}):")
        st.code(selected_sequence)

    elif file_type == "Text" and "text_data" in st.session_state:
        text_data = st.session_state["text_data"]
        sequence_idx = st.selectbox("Choose a sequence:", options=range(1, len(text_data["dna_sequences"]) + 1))
        selected_sequence = text_data["dna_sequences"][sequence_idx - 1]
        st.text(f"Selected Sequence (#{sequence_idx}):")
        st.code(selected_sequence)

    else:
        st.info("CSV data is not currently supported for algorithm selection.")
        return

    # Choose Algorithm
    algorithm = st.selectbox("Choose an algorithm to apply:", options=["Reverse Sequence", "Count Bases", "naive_match", "boyer_moore", "IndexSorted", "suffix_array", "overlap"])

    # Move input fields outside the main button condition
    if algorithm == "naive_match":
        target = st.text_input("Enter your pattern:")
        if st.button("Find Matches", key="naive_match_button"):
            if target and selected_sequence:
                st.text("Naive Match Result:")
                st.write("Matches found at indices:")
                matches = []
                for i in range(len(selected_sequence) - len(target) + 1):
                    if selected_sequence[i:i+len(target)] == target:
                        matches.append(i)
                if matches:
                    st.write(matches)
                else:
                    st.write("No matches found")
            else:
                st.warning("Please enter a pattern to search for")

    elif algorithm == "boyer_moore":
        target = st.text_input("Enter your pattern:", key="boyer_pattern")
        if st.button("Find Matches", key="boyer_button"):
            if target and selected_sequence:
                def build_bad_char_table(pattern):
                    bad_char = {}
                    for i in range(len(pattern)):
                        bad_char[pattern[i]] = i
                    return bad_char

                def boyer_moore_search(text, pattern):
                    matches = []
                    if not pattern or not text:
                        return matches
                    bad_char = build_bad_char_table(pattern)
                    m = len(pattern)
                    n = len(text)
                    i = 0
                    while i <= n - m:
                        j = m - 1
                        while j >= 0 and pattern[j] == text[i + j]:
                            j -= 1
                        if j < 0:
                            matches.append(i)
                            i += 1
                        else:
                            bad_char_shift = j - bad_char.get(text[i + j], -1)
                            i += max(1, bad_char_shift)
                    return matches

                matches = boyer_moore_search(selected_sequence, target)
                st.text("Boyer-Moore Result:")
                if matches:
                    st.write(f"Pattern found at indices: {matches}")
                else:
                    st.write("No matches found")
            else:
                st.warning("Please enter a pattern to search for")

    elif algorithm == "IndexSorted":
        k = st.number_input("Enter k-mer length:", min_value=1, 
                          max_value=len(selected_sequence) if selected_sequence else 10, 
                          value=3)
        if st.button("Generate k-mers", key="index_sorted_button"):
            if selected_sequence:
                kmers = []
                for i in range(len(selected_sequence) - k + 1):
                    kmer = selected_sequence[i:i+k]
                    kmers.append((kmer, i))
                kmers.sort()
                st.text("Sorted k-mers and their positions:")
                for kmer, pos in kmers:
                    st.write(f"k-mer: {kmer}, position: {pos}")

    elif algorithm == "suffix_array":
        if st.button("Generate Suffix Array", key="suffix_button"):
            if selected_sequence:
                suffixes = [(selected_sequence[i:], i) for i in range(len(selected_sequence))]
                suffixes.sort()
                suffix_array = [pos for (suffix, pos) in suffixes]
                st.text("Suffix Array Result:")
                st.write("Positions in sorted order:")
                st.write(suffix_array)
                st.text("Corresponding suffixes:")
                for pos in suffix_array:
                    st.write(f"{pos}: {selected_sequence[pos:]}")

    elif algorithm == "overlap":
        min_overlap = st.number_input("Minimum overlap length:", 
                                    min_value=1, value=3)
        other_sequence = st.text_area("Enter another sequence for overlap:")
        if st.button("Find Overlap", key="overlap_button"):
            if other_sequence and selected_sequence:
                def find_overlap(seq1, seq2, min_len):
                    for i in range(len(seq1), min_len-1, -1):
                        if seq2.startswith(seq1[-i:]):
                            return i
                    return 0

                overlap_length = find_overlap(selected_sequence, other_sequence, min_overlap)
                if overlap_length > 0:
                    st.text(f"Found overlap of length {overlap_length}")
                    st.text("Overlapping region:")
                    st.code(selected_sequence[-overlap_length:])
                else:
                    st.write(f"No overlap of length ≥{min_overlap} found")
            else:
                st.warning("Please enter a sequence for overlap comparison")

    # Handle simple algorithms with a single button
    elif algorithm in ["Reverse Sequence", "Count Bases"]:
        if st.button("Apply Algorithm", key="simple_algorithm"):
            if not selected_sequence:
                st.error("Please select a sequence first!")
                return

            if algorithm == "Reverse Sequence":
                result = selected_sequence[::-1]
                st.text("Reversed Sequence:")
                st.code(result)
                
            elif algorithm == "Count Bases":
                counts = {base: selected_sequence.count(base) for base in "ACGT"}
                st.text("Base Counts:")
                st.json(counts)
                
# Function for Contact Us page
def contact_page():
    st.title("Contact Us")
    st.write("If you have any questions or feedback, feel free to reach out to us.")

    # Contact Form
    name = st.text_input("Your Name")
    email = st.text_input("Your Email")
    message = st.text_area("Your Message")

    # Button and form validation
    if st.button("Submit"):
        if name and email and message:
            st.success(f"Thank you, {name}, for your message! We'll get back to you soon.")
        else:
            st.error("Please fill out all the fields before submitting.")


if page == "Home":
    home_page()
elif page == "Data upload":
    data_upload_page()
elif page == "Apply an Algorithm":
    algorithm_apply()
elif page == "Apply for pattern":
    apply_for_pattern()
elif page == "Contact Us":
    contact_page()