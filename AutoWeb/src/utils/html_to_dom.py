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
    

# sample_html = """
# <div>
#     <button id="btn1" onclick="alert('Clicked!')">Click Me</button>
#     <a href="https://example.com" role="link">Visit Example</a>
#     <input type="text" placeholder="Enter name" />
#     <div role="button" tabindex="0">Custom Button</div>
#     <div>
#         <span role="checkbox" aria-checked="false">Option 1</span>
#         <span role="checkbox" aria-checked="true">Option 2</span>
#     </div>
#     <span backend_node_id="123">Annotated Element</span>
# </div>
# """

# elements =extract_interactive_elements(sample_html)

# print("Extracted Interactive Elements:", elements)