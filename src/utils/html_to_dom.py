from lxml import html

def extract_interactive_elements(html_string: str):
    """
    Extract interactive DOM elements similar to SeeAct.
    Covers standard interactive tags plus elements annotated with
    common ARIA roles and Mind2Web's backend_node_id attribute.
    """
    tree = html.fromstring(html_string)

    # Broad interactive-element selector:
    #   • Standard interactive tags
    #   • Elements with common interactive ARIA roles
    #   • Elements with tabindex (keyboard-navigable)
    #   • Elements with onclick handlers
    #   • Elements annotated by Mind2Web with backend_node_id
    xpath_query = """
    //button | 
    //a | 
    //input | 
    //select | 
    //textarea |
    //label |
    //*[@role='button'] |
    //*[@role='tab'] |
    //*[@role='menuitem'] |
    //*[@role='option'] |
    //*[@role='link'] |
    //*[@role='checkbox'] |
    //*[@role='radio'] |
    //*[@role='switch'] |
    //*[@role='combobox'] |
    //*[@role='listbox'] |
    //*[@tabindex] |
    //*[@onclick] |
    //*[@backend_node_id]
    """

    elements = tree.xpath(xpath_query)

    # Deduplicate: xpath union (|) can return the same element via multiple
    # matching rules.  Use element identity (id()) to keep unique elements.
    seen_ids = set()
    unique_elements = []
    for el in elements:
        eid = id(el)
        if eid not in seen_ids:
            seen_ids.add(eid)
            unique_elements.append(el)

    structured_elements = []

    for idx, el in enumerate(unique_elements):
        text = el.text_content().strip()
        tag = el.tag
        attrs = dict(el.attrib)          # convert lxml _Attrib → plain dict

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
    