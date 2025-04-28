import uuid
import streamlit as st
import requests
import os
import time
from PIL import Image
import io
from dotenv import load_dotenv

load_dotenv()


FASTAPI_URL = os.getenv("FASTAPI_URL")
FASTAPI_URL="http://localhost:8000"
def search_page():
    st.title("🔍 Find Similar Clothing Items")

    # Add search type selection
    search_type = st.radio("Search by:", ["Image", "Text"], horizontal=True)
    
    if search_type == "Image":
        img_bytes = None
        enable = st.checkbox("Enable camera")
        img_bytes = st.camera_input("Capture an item to search", 
                                    key="mobile_cam",
                                    label_visibility="collapsed", disabled=not enable)
        
        comment = st.text_input("Additional search description (optional)", 
                              placeholder="e.g., 'stripes' or 't-shirt'")
        
        if img_bytes and st.button("Search Similar Items"):
            with st.spinner("Finding similar items..."):
                try:
                    search_response = requests.post(
                        f"{FASTAPI_URL}/search",
                        files={"image": ("query.jpg", img_bytes, "image/jpeg")},
                        data={"comment": comment}
                    )
                    
                    if search_response.status_code == 200:
                        results = search_response.json()
                        if not results:
                            st.warning("No similar items found")
                            return
                        
                        st.subheader("Top Matching Items")
                        cols = st.columns(3)
                        for idx, res in enumerate(results[:3]):
                            with cols[idx % 3]:
                                st.image(f"{FASTAPI_URL}/images/{res['filename']}")
                                tags_display = ", ".join(res.get("tags", []))
                                st.caption(f"💬 {res['comment']} | Tags: {tags_display}")
                    else:
                        st.error("Search failed. Please try again.")
                        
                except requests.ConnectionError:
                    st.error("Could not connect to search server")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    else:  # Text-based search
        search_text = st.text_input("Describe the item you're looking for", 
                                  placeholder="e.g., 'red striped t-shirt' or 'blue jeans'")
        
        if search_text and st.button("Search by Description"):
            with st.spinner("Finding matching items..."):
                try:
                    search_response = requests.post(
                        f"{FASTAPI_URL}/search-by-text",
                        data={"text": search_text}
                    )
                    
                    if search_response.status_code == 200:
                        results = search_response.json()
                        if not results:
                            st.warning("No matching items found")
                            return
                        
                        st.subheader("Top Matching Items")
                        cols = st.columns(3)
                        for idx, res in enumerate(results[:3]):
                            with cols[idx % 3]:
                                st.image(f"{FASTAPI_URL}/images/{res['filename']}")
                                tags_display = ", ".join(res.get("tags", []))
                                st.caption(f"💬 {res['comment']} | Tags: {tags_display}")
                    else:
                        st.error("Search failed. Please try again.")
                        
                except requests.ConnectionError:
                    st.error("Could not connect to search server")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

def upload_page():
    st.title("📤 Add New Items")
    
    with st.form("add_item_form"):
        st.subheader("Add New Clothing Item")
        new_file = st.file_uploader("Upload clothing image", 
                                  type=["jpg", "jpeg", "png"],
                                  key="add_item_upload")
        new_comment = st.text_input("Item description (required)", 
                                  placeholder="Describe key features")
        new_tags = st.text_input("Tags (comma-separated)", 
                               placeholder="e.g., casual, cotton, summer")
        submitted = st.form_submit_button("Add to Collection")
        
        if submitted:
            if new_file and new_comment:
                with st.spinner("Adding item to wardrobe..."):
                    try:
                        # Generate unique filename with short UUID
                        original_name = new_file.name
                        short_uuid = uuid.uuid4().hex[:8]  # 8-character UUID
                        name_part, ext = os.path.splitext(original_name)
                        unique_filename = f"{name_part}_{short_uuid}{ext}"
                        
                        response = requests.post(
                            f"{FASTAPI_URL}/add-item",
                            files={"image": (unique_filename, new_file.getvalue(), new_file.type)},
                            data={
                                "comment": new_comment,
                                "tags": new_tags  # Add tags to request
                            }
                        )
                        if response.status_code == 200:
                            st.success("✅ Item added successfully!")
                            st.session_state.show_rebuild = True
                        else:
                            st.error(f"❌ Add failed: {response.json().get('detail', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"🚨 Connection error: {str(e)}")
            else:
                st.warning("⚠️ Please upload an image and provide a description")

    if st.session_state.get("show_rebuild", False):
        st.markdown("---")
        if st.button("🔄 Rebuild Entire Search Index", 
                   help="Update search database with new items"):
            with st.spinner("Rebuilding index - this may take a while..."):
                try:
                    response = requests.post(f"{FASTAPI_URL}/rebuild-index")
                    if response.status_code == 200:
                        st.success("✨ Index rebuilt successfully!")
                        st.session_state.show_rebuild = False
                    else:
                        st.error("❌ Index rebuild failed")
                except Exception as e:
                    st.error(f"🚨 Connection error: {str(e)}")

def browse_page():
    st.title("📚 Browse Clothing Collection")
    
    # Fetch all items from backend
    try:
        response = requests.get(f"{FASTAPI_URL}/items")
        if response.status_code == 200:
            all_items = response.json()
        else:
            st.error("Failed to load items")
            all_items = []
    except Exception as e:
        st.error(f"Connection error: {str(e)}")
        all_items = []

    # Tag filtering
    all_tags = sorted({tag for item in all_items for tag in item.get('tags', [])})
    selected_tags = st.multiselect("Filter by tags", all_tags)

    # Apply tag filters
    filtered_items = all_items
    if selected_tags:
        filtered_items = [item for item in all_items 
                         if any(tag in selected_tags for tag in item.get('tags', []))]

    # Sorting options
    sort_option = st.selectbox("Sort by", 
                              ["Newest First", "Oldest First", "Comment (A-Z)", "Comment (Z-A)"])
    
    if sort_option == "Newest First":
        filtered_items = filtered_items[::-1]  # Reverse since new items are appended
    elif sort_option == "Comment (A-Z)":
        filtered_items.sort(key=lambda x: x.get('comment', '').lower())
    elif sort_option == "Comment (Z-A)":
        filtered_items.sort(key=lambda x: x.get('comment', '').lower(), reverse=True)

    # Display grid
    st.markdown(f"**Showing {len(filtered_items)} items**")
    cols = st.columns(3)
    
    for idx, item in enumerate(filtered_items):
        with cols[idx % 3]:
            # Image display
            try:
                st.image(f"{FASTAPI_URL}/images/{item['filename']}", 
                        use_container_width=True)
            except:
                st.error("Image not found")
            
            # Metadata
            st.caption(f"💬 {item.get('comment', '')}")
            tags_display = ", ".join(item.get('tags', []))
            if tags_display:
                st.write(f"🏷️ `{tags_display}`")
            
            # Delete button
            if st.button("Delete", key=f"delete_{item['filename']}"):
                try:
                    delete_response = requests.delete(
                        f"{FASTAPI_URL}/items/{item['filename']}"
                    )
                    if delete_response.status_code == 200:
                        st.success("Item deleted successfully")
                        time.sleep(1)
                        st.experimental_rerun()
                    else:
                        st.error("Failed to delete item")
                except Exception as e:
                    st.error(f"Delete error: {str(e)}")

def outfit_page():
    st.title("👗 Smart Outfit Matching")
    
    # Outfit input
    outfit_img = st.file_uploader("Upload outfit inspiration", type=["jpg", "jpeg", "png"])
    analysis_done = False  # Track analysis state
    
    if outfit_img:
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(outfit_img, caption="Your Style Inspiration")
            
        with col2:
            # Only analyze when button is clicked
            if st.button("Analyze Outfit Components"):
                with st.spinner("Analyzing outfit components..."):
                    try:
                        response = requests.post(
                            f"{FASTAPI_URL}/analyze-outfit",
                            files={"image": ("outfit.jpg", outfit_img.getvalue(), "image/jpeg")}
                        )
                        
                        if response.status_code == 200:
                            st.session_state.outfit_components = response.json()
                            analysis_done = True
                        else:
                            st.error("Analysis failed. Please try again.")
                            
                    except Exception as e:
                        st.error(f"Connection error: {str(e)}")

        # Show analysis results if available
        if analysis_done or "outfit_components" in st.session_state:
            components = st.session_state.outfit_components
            st.subheader("Detected Components")
            
            seg_cols = st.columns(2)
            for idx, comp in enumerate(components):
                with seg_cols[idx % 2]:
                    # Display component from backend URL
                    st.image(
                        f"{FASTAPI_URL}/{comp['segment_path']}",
                        caption=f"{comp['category']} ({comp['confidence']*100:.1f}%)"
                    )
                    
                    if st.button(f"Find similar {comp['category']}", key=f"search_{comp['segment_path']}"):
                        with st.spinner(f"Finding similar {comp['category']}..."):
                            try:
                                # Get segmented image from backend
                                segment_response = requests.get(
                                    f"{FASTAPI_URL}/{comp['segment_path']}"
                                )
                                
                                if segment_response.status_code == 200:
                                    search_response = requests.post(
                                        f"{FASTAPI_URL}/search",
                                        files={"image": (comp['segment_path'], segment_response.content, "image/jpeg")},
                                        data={"comment": comp['category']}
                                    )
                                    
                                    if search_response.status_code == 200:
                                        show_results(search_response.json(), comp['category'])
                                    else:
                                        st.error("Search request failed")
                                else:
                                    st.error("Could not fetch segmented image")
                                    
                            except Exception as e:
                                st.error(f"Search error: {str(e)}")


def show_results(items, category):
    """Display search results in a grid format"""
    if not items:
        st.warning(f"No matching {category} found")
        return
    
    st.subheader(f"Top {category.capitalize()} Matches")
    cols = st.columns(3)
    
    for idx, item in enumerate(items[:3]):
        with cols[idx % 3]:
            try:
                st.image(
                    f"{FASTAPI_URL}/images/{item['filename']}",
                    use_container_width=True,
                    caption=item.get('comment', '')
                )
                tags_display = ", ".join(item.get('tags', []))
                if tags_display:
                    st.write(f"🏷️ `{tags_display}`")
            except Exception as e:
                st.error(f"Error displaying item: {str(e)}")
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Browse", "Search", "Upload", "Outfit"])
    st.sidebar.markdown("---")
    st.sidebar.info(
        "Browse your complete clothing collection. Filter by tags, sort items, "
        "manage your wardobe, and recommend by outfit."
    )
    
    if page == "Browse":
        browse_page()
    elif page == "Search":
        search_page()
    elif page == "Upload":
        upload_page()
    elif page == "Outfit":
        outfit_page()

if __name__ == "__main__":
    main()
