from lxml import html

def extract_interactive_elements(html_string: str):
    """
    Extract interactive DOM elements similar to SeeAct.
    """
    tree = html.fromstring(html_string)

    # Common interactive tags
    xpath_query = """
    //button | 
    //a | 
    //input | 
    //select | 
    //textarea |
    //*[@role='button'] |
    //*[@onclick]
    """

    elements = tree.xpath(xpath_query)

    structured_elements = []

    for idx, el in enumerate(elements):
        text = el.text_content().strip()
        tag = el.tag
        attrs = el.attrib

        # Create textual representation
        element_repr = f"<{tag}> text='{text}' attrs={attrs}"

        structured_elements.append({
            "index": idx,
            "tag": tag,
            "text": text,
            "attributes": attrs,
            "representation": element_repr
        })

    return structured_elements
    